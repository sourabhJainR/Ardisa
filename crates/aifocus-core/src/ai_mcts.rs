//! Bounded deterministic Monte Carlo tree-search storage primitives.
//!
//! This is the runtime foundation for issue #171. It is not, by itself, a
//! claim that MCTSTree is a supported .ardisa source construct; that requires
//! semantic, typed-IR, native-instruction, artifact-verifier, and VM wiring.

pub const MAX_MCTS_NODES: usize = 4_096;
pub const MAX_MCTS_BRANCHES: usize = 64;
pub const MAX_MCTS_DEPTH: usize = 256;
pub const MAX_MCTS_REWARD_ABS: f64 = 1_000_000.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MctsError {
    InvalidCapacity,
    NodeLimit,
    BranchLimit,
    InvalidNode,
    DuplicateChildState,
    InvalidReward,
    DepthLimit,
    CorruptTree,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MctsNode {
    pub state_id: u64,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub visits: u64,
    pub value_sum: f64,
}

impl MctsNode {
    pub fn mean_value(&self) -> f64 {
        if self.visits == 0 { 0.0 } else { self.value_sum / self.visits as f64 }
    }
}

/// Arena-indexed MCTS tree. Node indices remain stable for the tree's lifetime;
/// this first subset deliberately has no compaction or deletion operation.
#[derive(Debug, Clone, PartialEq)]
pub struct MctsTree {
    max_nodes: usize,
    max_branches: usize,
    nodes: Vec<MctsNode>,
}

impl MctsTree {
    pub fn new(max_nodes: usize, max_branches: usize) -> Result<Self, MctsError> {
        if max_nodes == 0 || max_nodes > MAX_MCTS_NODES ||
            max_branches == 0 || max_branches > MAX_MCTS_BRANCHES {
            return Err(MctsError::InvalidCapacity);
        }
        Ok(Self { max_nodes, max_branches, nodes: Vec::new() })
    }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }
    pub fn nodes(&self) -> &[MctsNode] { &self.nodes }

    pub fn create_root(&mut self, state_id: u64) -> Result<usize, MctsError> {
        if !self.nodes.is_empty() { return Err(MctsError::CorruptTree); }
        if self.max_nodes == 0 { return Err(MctsError::NodeLimit); }
        self.nodes.push(MctsNode { state_id, parent: None, children: Vec::new(), visits: 0, value_sum: 0.0 });
        Ok(0)
    }

    pub fn root(&self) -> Result<usize, MctsError> {
        if self.nodes.first().is_some_and(|node| node.parent.is_none()) { Ok(0) }
        else { Err(MctsError::InvalidNode) }
    }

    pub fn child(&self, parent: usize, ordinal: usize) -> Result<usize, MctsError> {
        self.nodes.get(parent).ok_or(MctsError::InvalidNode)?
            .children.get(ordinal).copied().ok_or(MctsError::InvalidNode)
    }

    pub fn expand(&mut self, parent: usize, state_id: u64) -> Result<usize, MctsError> {
        let p = self.nodes.get(parent).ok_or(MctsError::InvalidNode)?;
        if p.children.len() >= self.max_branches { return Err(MctsError::BranchLimit); }
        if self.nodes.len() >= self.max_nodes { return Err(MctsError::NodeLimit); }
        if p.children.iter().any(|&child| self.nodes.get(child).is_some_and(|n| n.state_id == state_id)) {
            return Err(MctsError::DuplicateChildState);
        }
        let depth = self.depth_of(parent)?;
        if depth + 1 >= MAX_MCTS_DEPTH { return Err(MctsError::DepthLimit); }
        let child = self.nodes.len();
        // All validation precedes either mutation: expansion is atomic on error.
        self.nodes.push(MctsNode { state_id, parent: Some(parent), children: Vec::new(), visits: 0, value_sum: 0.0 });
        self.nodes[parent].children.push(child);
        Ok(child)
    }

    pub fn update(&mut self, node: usize, reward: f64) -> Result<(), MctsError> {
        validate_reward(reward)?;
        let target = self.nodes.get(node).ok_or(MctsError::InvalidNode)?;
        if target.visits == u64::MAX { return Err(MctsError::InvalidReward); }
        let next_sum = target.value_sum + reward;
        if !next_sum.is_finite() || next_sum.abs() > MAX_MCTS_REWARD_ABS * (target.visits.saturating_add(1) as f64) {
            return Err(MctsError::InvalidReward);
        }
        self.nodes[node].visits += 1;
        self.nodes[node].value_sum = next_sum;
        Ok(())
    }

    /// Update a node and every ancestor, root first validation then mutation.
    /// The full parent chain is checked before any visit/value is changed.
    pub fn backpropagate(&mut self, node: usize, reward: f64) -> Result<(), MctsError> {
        validate_reward(reward)?;
        let mut path = Vec::new();
        let mut cursor = node;
        loop {
            let current = self.nodes.get(cursor).ok_or(MctsError::InvalidNode)?;
            if current.visits == u64::MAX { return Err(MctsError::InvalidReward); }
            let next_sum = current.value_sum + reward;
            if !next_sum.is_finite() || next_sum.abs() > MAX_MCTS_REWARD_ABS * (current.visits.saturating_add(1) as f64) {
                return Err(MctsError::InvalidReward);
            }
            path.push(cursor);
            match current.parent {
                Some(parent) => {
                    if parent >= cursor || !self.nodes.get(parent).is_some_and(|p| p.children.contains(&cursor)) {
                        return Err(MctsError::CorruptTree);
                    }
                    if path.len() >= MAX_MCTS_DEPTH { return Err(MctsError::DepthLimit); }
                    cursor = parent;
                }
                None => break,
            }
        }
        for index in path {
            self.nodes[index].visits += 1;
            self.nodes[index].value_sum += reward;
        }
        Ok(())
    }

    fn depth_of(&self, node: usize) -> Result<usize, MctsError> {
        let mut cursor = node;
        let mut depth = 0;
        loop {
            let current = self.nodes.get(cursor).ok_or(MctsError::InvalidNode)?;
            match current.parent {
                Some(parent) => {
                    if parent >= cursor || !self.nodes.get(parent).is_some_and(|p| p.children.contains(&cursor)) {
                        return Err(MctsError::CorruptTree);
                    }
                    depth += 1;
                    if depth >= MAX_MCTS_DEPTH { return Err(MctsError::DepthLimit); }
                    cursor = parent;
                }
                None => return Ok(depth),
            }
        }
    }
}

fn validate_reward(reward: f64) -> Result<(), MctsError> {
    if !reward.is_finite() || reward.abs() > MAX_MCTS_REWARD_ABS {
        Err(MctsError::InvalidReward)
    } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_root_expands_and_finds_children_deterministically() {
        let mut tree = MctsTree::new(8, 2).unwrap();
        assert_eq!(tree.create_root(10), Ok(0));
        let a = tree.expand(0, 11).unwrap();
        let b = tree.expand(0, 12).unwrap();
        assert_eq!((a, b), (1, 2));
        assert_eq!(tree.root(), Ok(0));
        assert_eq!(tree.child(0, 0), Ok(a));
        assert_eq!(tree.child(0, 1), Ok(b));
        assert_eq!(tree.nodes()[a].parent, Some(0));
    }

    #[test]
    fn backpropagates_to_root_and_tracks_mean_value() {
        let mut tree = MctsTree::new(8, 4).unwrap();
        tree.create_root(1).unwrap();
        let child = tree.expand(0, 2).unwrap();
        let leaf = tree.expand(child, 3).unwrap();
        tree.backpropagate(leaf, 0.75).unwrap();
        assert_eq!(tree.nodes()[leaf].visits, 1);
        assert_eq!(tree.nodes()[child].visits, 1);
        assert_eq!(tree.nodes()[0].visits, 1);
        assert_eq!(tree.nodes()[0].mean_value(), 0.75);
    }

    #[test]
    fn rejects_bounds_duplicate_edges_invalid_indices_and_non_finite_rewards() {
        assert_eq!(MctsTree::new(0, 1), Err(MctsError::InvalidCapacity));
        let mut tree = MctsTree::new(2, 1).unwrap();
        tree.create_root(1).unwrap();
        assert_eq!(tree.expand(0, 2), Ok(1));
        assert_eq!(tree.expand(0, 2), Err(MctsError::BranchLimit));
        assert_eq!(tree.expand(0, 3), Err(MctsError::BranchLimit));
        assert_eq!(tree.expand(9, 3), Err(MctsError::InvalidNode));
        assert_eq!(tree.child(0, 9), Err(MctsError::InvalidNode));
        assert_eq!(tree.update(1, f64::NAN), Err(MctsError::InvalidReward));
        assert_eq!(tree.backpropagate(1, f64::INFINITY), Err(MctsError::InvalidReward));
        assert_eq!(tree.nodes()[0].visits, 0);
        assert_eq!(tree.nodes()[1].visits, 0);
    }

    #[test]
    fn mutation_is_atomic_for_invalid_reward() {
        let mut tree = MctsTree::new(8, 2).unwrap();
        tree.create_root(1).unwrap();
        let child = tree.expand(0, 2).unwrap();
        let before = tree.clone();
        assert_eq!(tree.backpropagate(child, MAX_MCTS_REWARD_ABS + 1.0), Err(MctsError::InvalidReward));
        assert_eq!(tree, before);
    }

    #[test]
    fn root_is_unique_and_depth_is_bounded_by_global_ceiling() {
        let mut tree = MctsTree::new(8, 2).unwrap();
        tree.create_root(1).unwrap();
        assert_eq!(tree.create_root(2), Err(MctsError::CorruptTree));
        assert_eq!(tree.expand(99, 2), Err(MctsError::InvalidNode));
    }
}
