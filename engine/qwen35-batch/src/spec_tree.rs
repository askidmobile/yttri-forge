//! Trie-представление спекулятивного дерева.
//!
//! Это первый шаг к однопроходной проверке ветвящегося драфта:
//! узлы хранят токен, родителя и диапазон поддерева в линейном порядке.
//! `FlatTrie` используется для передачи дерева в CUDA и построения маски внимания.

#[derive(Debug, Clone, PartialEq)]
pub struct TrieNode {
    pub token: u32,
    pub parent: Option<usize>,
    pub logprob: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlatTrieNode {
    pub node: TrieNode,
    pub start: usize,
    pub end: usize,
    pub depth: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlatTrie {
    nodes: Vec<FlatTrieNode>,
}

impl FlatTrie {
    pub fn from_edges(edges: &[(usize, u32, f32)]) -> Self {
        let mut nodes = Vec::with_capacity(edges.len() + 1);
        nodes.push(FlatTrieNode {
            node: TrieNode { token: 0, parent: None, logprob: 0.0 },
            start: 0,
            end: 0,
            depth: 0,
        });

        for &(parent, token, logprob) in edges {
            let idx = nodes.len();
            nodes.push(FlatTrieNode {
                node: TrieNode { token, parent: Some(parent), logprob },
                start: idx,
                end: idx,
                depth: nodes[parent].depth + 1,
            });
        }

        // Пересчитать диапазоны поддеревьев после добавления всех узлов.
        let mut ranges = vec![(0usize, 0usize); nodes.len()];
        for i in (0..nodes.len()).rev() {
            let mut start = i;
            let mut end = i;
            for j in 0..nodes.len() {
                if nodes[j].node.parent == Some(i) {
                    start = start.min(ranges[j].0);
                    end = end.max(ranges[j].1);
                }
            }
            ranges[i] = (start, end);
        }
        for (i, range) in ranges.into_iter().enumerate() {
            nodes[i].start = range.0;
            nodes[i].end = range.1;
        }

        Self { nodes }
    }

    pub fn nodes(&self) -> &[FlatTrieNode] {
        &self.nodes
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_tree_ranges() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (1, 2, -2.0), (2, 3, -3.0)]);
        assert_eq!(trie.len(), 4);
        let nodes = trie.nodes();
        assert_eq!(nodes[0].start, 0);
        assert_eq!(nodes[0].end, 3);
        assert_eq!(nodes[1].start, 1);
        assert_eq!(nodes[1].end, 3);
        assert_eq!(nodes[2].start, 2);
        assert_eq!(nodes[2].end, 3);
        assert_eq!(nodes[3].start, 3);
        assert_eq!(nodes[3].end, 3);
    }

    #[test]
    fn branching_tree_ranges() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (0, 2, -2.0), (1, 3, -3.0), (2, 4, -4.0)]);
        assert_eq!(trie.len(), 5);
        let nodes = trie.nodes();
        assert_eq!(nodes[0].start, 0);
        assert_eq!(nodes[0].end, 4);
        assert_eq!(nodes[1].start, 1);
        assert_eq!(nodes[1].end, 3);
        assert_eq!(nodes[2].start, 2);
        assert_eq!(nodes[2].end, 4);
        assert_eq!(nodes[3].start, 3);
        assert_eq!(nodes[3].end, 3);
        assert_eq!(nodes[4].start, 4);
        assert_eq!(nodes[4].end, 4);
    }
}
