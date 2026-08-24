#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>

static std::vector<llama_token> read_tokens(const char * path) {
    std::ifstream input(path);
    std::vector<llama_token> tokens;
    long long token = 0;
    while (input >> token) {
        if (token < 0 || token > std::numeric_limits<llama_token>::max()) {
            std::fprintf(stderr, "invalid token in %s: %lld\n", path, token);
            return {};
        }
        tokens.push_back(static_cast<llama_token>(token));
    }
    if (!input.eof()) {
        std::fprintf(stderr, "cannot parse token file: %s\n", path);
        return {};
    }
    return tokens;
}

static std::set<int> read_steps(const char * text) {
    std::set<int> steps;
    std::string value(text);
    size_t begin = 0;
    while (begin < value.size()) {
        const size_t end = value.find(',', begin);
        const std::string item = value.substr(begin, end - begin);
        char * tail = nullptr;
        const long step = std::strtol(item.c_str(), &tail, 10);
        if (item.empty() || *tail != '\0' || step < 0 || step > std::numeric_limits<int>::max()) {
            return {};
        }
        steps.insert(static_cast<int>(step));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return steps;
}

static llama_token argmax(const float * logits, int32_t n_vocab) {
    llama_token best = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (llama_token token = 0; token < n_vocab; ++token) {
        if (logits[token] > best_value) {
            best = token;
            best_value = logits[token];
        }
    }
    return best;
}

static void print_float(float value) {
    if (std::isfinite(value)) {
        std::printf("%.9g", value);
    } else {
        std::printf("null");
    }
}

static void print_ids(const std::vector<llama_token> & ids) {
    std::putchar('[');
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            std::putchar(',');
        }
        std::printf("%d", ids[i]);
    }
    std::putchar(']');
}

static void print_record(int step, const float * logits, int32_t n_vocab, bool include_values) {
    std::vector<int32_t> indices(n_vocab);
    std::iota(indices.begin(), indices.end(), 0);
    const int32_t top_n = std::min<int32_t>(10, n_vocab);
    std::partial_sort(indices.begin(), indices.begin() + top_n, indices.end(), [&](int32_t a, int32_t b) {
        if (logits[a] == logits[b]) {
            return a < b;
        }
        return logits[a] > logits[b];
    });

    double sum = 0.0;
    double sum_sq = 0.0;
    int32_t finite = 0;
    uint64_t checksum = UINT64_C(0xcbf29ce484222325);
    for (int32_t i = 0; i < n_vocab; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, logits + i, sizeof(bits));
        checksum ^= bits;
        checksum *= UINT64_C(0x100000001b3);
        if (std::isfinite(logits[i])) {
            ++finite;
            const double value = logits[i];
            sum += value;
            sum_sq += value * value;
        }
    }

    const float margin = top_n >= 2 ? logits[indices[0]] - logits[indices[1]] : std::numeric_limits<float>::quiet_NaN();
    std::printf("{\"type\":\"logits\",\"step\":%d,\"argmax\":%d,\"margin\":", step, indices[0]);
    print_float(margin);
    std::printf(",\"finite\":%d,\"sum\":%.17g,\"sum_sq\":%.17g,\"checksum\":\"%016llx\",\"top\":[", finite, sum, sum_sq, static_cast<unsigned long long>(checksum));
    for (int32_t i = 0; i < top_n; ++i) {
        if (i != 0) {
            std::putchar(',');
        }
        std::printf("{\"id\":%d,\"logit\":", indices[i]);
        print_float(logits[indices[i]]);
        std::putchar('}');
    }
    std::putchar(']');
    if (include_values) {
        std::printf(",\"values\":[");
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (i != 0) {
                std::putchar(',');
            }
            print_float(logits[i]);
        }
        std::putchar(']');
    }
    std::printf("}\n");
}

static void quiet_log(enum ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: llama-logits <model.gguf> <prompt-tokens.txt> <forced-tokens.txt|-> <steps> [full-steps]\n");
        return 2;
    }
    const int steps = std::atoi(argv[4]);
    const auto full_steps = argc == 6 ? read_steps(argv[5]) : std::set<int>{};
    if (argc == 6 && full_steps.empty()) {
        std::fprintf(stderr, "full-steps must be comma-separated non-negative integers\n");
        return 2;
    }
    auto prompt = read_tokens(argv[2]);
    const bool free_run = std::strcmp(argv[3], "-") == 0;
    auto forced = free_run ? std::vector<llama_token>{} : read_tokens(argv[3]);
    if (prompt.empty() || steps <= 0 || (!free_run && static_cast<int>(forced.size()) < steps)) {
        std::fprintf(stderr, "need non-empty prompt and at least %d forced tokens, or '-' for greedy\n", steps);
        return 2;
    }

    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    const auto load_started = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "cannot load model: %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = std::max<uint32_t>(512, static_cast<uint32_t>(prompt.size() + steps + 8));
    ctx_params.n_batch = static_cast<uint32_t>(prompt.size());
    ctx_params.n_ubatch = static_cast<uint32_t>(prompt.size());
    ctx_params.n_seq_max = 1;
    ctx_params.n_outputs_max = 1;
    ctx_params.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "cannot create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const auto model_load = std::chrono::steady_clock::now() - load_started;

    std::printf("{\"type\":\"run\",\"schema\":\"qwen35moe-logits-v1\",\"backend\":\"llama.cpp@8e7f22b\",\"prompt_tokens\":");
    print_ids(prompt);
    std::printf(",\"steps\":%d}\n", steps);

    llama_batch batch = llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()));
    const auto prefill_started = std::chrono::steady_clock::now();
    int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        std::fprintf(stderr, "prompt decode failed: %d\n", rc);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    llama_synchronize(ctx);
    const auto prefill_time = std::chrono::steady_clock::now() - prefill_started;

    std::vector<llama_token> predicted;
    std::vector<llama_token> fed;
    predicted.reserve(steps);
    fed.reserve(steps);
    std::chrono::steady_clock::duration decode_time{};
    int decode_calls = 0;
    for (int step = 0; step < steps; ++step) {
        float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "missing logits at step %d\n", step);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const llama_token prediction = argmax(logits, n_vocab);
        print_record(step, logits, n_vocab, full_steps.count(step) != 0);
        predicted.push_back(prediction);
        llama_token token = free_run ? prediction : forced[step];
        fed.push_back(token);
        if (step + 1 == steps) {
            break;
        }
        batch = llama_batch_get_one(&token, 1);
        const auto decode_started = std::chrono::steady_clock::now();
        rc = llama_decode(ctx, batch);
        if (rc != 0) {
            std::fprintf(stderr, "token decode failed at step %d: %d\n", step, rc);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        llama_synchronize(ctx);
        decode_time += std::chrono::steady_clock::now() - decode_started;
        ++decode_calls;
    }

    std::printf("{\"type\":\"tokens\",\"ids\":");
    print_ids(predicted);
    std::printf(",\"fed_ids\":");
    print_ids(fed);
    std::printf("}\n");
    const double load_ms = std::chrono::duration<double, std::milli>(model_load).count();
    const double prefill_ms = std::chrono::duration<double, std::milli>(prefill_time).count();
    const double decode_ms = std::chrono::duration<double, std::milli>(decode_time).count();
    const double decode_tokens_per_s = decode_calls == 0 ? 0.0 : decode_calls * 1000.0 / decode_ms;
    std::printf(
        "{\"type\":\"performance\",\"model_load_ms\":%.6f,\"prefill_ms\":%.6f,"
        "\"prompt_tokens\":%zu,\"prefill_tokens_per_s\":%.6f,\"decode_ms\":%.6f,"
        "\"decode_calls\":%d,\"decode_tokens_per_s\":%.6f}\n",
        load_ms,
        prefill_ms,
        prompt.size(),
        prompt.size() * 1000.0 / prefill_ms,
        decode_ms,
        decode_calls,
        decode_tokens_per_s);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
