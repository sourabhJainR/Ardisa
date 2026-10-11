use std::collections::{HashMap, HashSet};

use crate::ast::{Block, Expr, ExprKind, Item, Module, StmtKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectKind {
    Read,
    Write,
    Call,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FunctionEffects {
    pub effects: HashSet<EffectKind>,
    pub calls: HashSet<String>,
    pub reads: HashSet<String>,
    pub writes: HashSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EffectModel {
    pub functions: HashMap<String, FunctionEffects>,
    pub dependencies: HashMap<String, HashSet<String>>,
}

/// Assignment statements contribute explicit write effects.
pub fn analyze(module: &Module) -> EffectModel {
    let mut functions = HashMap::new();
    for item in &module.items {
        let Item::Function(function) = item;
        let mut effects = FunctionEffects::default();
        collect_block(&function.body, &mut effects);
        functions.insert(function.name.clone(), effects);
    }

    // Propagate call effects to a fixed point so callers inherit the effects of
    // functions declared anywhere in the module.
    for _ in 0..module.items.len().max(1) {
        let before = functions.clone();
        let snapshot = functions.clone();
        for effects in functions.values_mut() {
            for callee in effects.calls.clone() {
                if let Some(callee_effects) = snapshot.get(&callee) {
                    effects
                        .effects
                        .extend(callee_effects.effects.iter().copied());
                    effects.reads.extend(callee_effects.reads.iter().cloned());
                    effects.writes.extend(callee_effects.writes.iter().cloned());
                }
            }
        }
        if functions == before {
            break;
        }
    }

    let dependencies = functions
        .iter()
        .map(|(name, effects)| (name.clone(), effects.calls.clone()))
        .collect();
    EffectModel {
        functions,
        dependencies,
    }
}

impl EffectModel {
    /// Return functions whose observed reads or writes touch a changed resource,
    /// including transitive callers. This is the basis for precise incremental
    /// invalidation without treating every source edit as a full rebuild.
    pub fn invalidated_by_resource(&self, resource: &str) -> HashSet<String> {
        let directly_affected = self
            .functions
            .iter()
            .filter_map(|(name, effects)| {
                (effects.reads.contains(resource) || effects.writes.contains(resource))
                    .then_some(name.clone())
            })
            .collect::<HashSet<_>>();
        let mut affected = directly_affected.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for (caller, dependencies) in &self.dependencies {
                if !affected.is_disjoint(dependencies) && affected.insert(caller.clone()) {
                    changed = true;
                }
            }
        }
        affected
    }

    pub fn depends_on(&self, caller: &str, callee: &str) -> bool {
        if caller == callee {
            return true;
        }
        let mut pending = vec![caller.to_string()];
        let mut seen = HashSet::new();
        while let Some(current) = pending.pop() {
            if !seen.insert(current.clone()) {
                continue;
            }
            for next in self.dependencies.get(&current).into_iter().flatten() {
                if next == callee {
                    return true;
                }
                pending.push(next.clone());
            }
        }
        false
    }
}

fn collect_block(block: &Block, effects: &mut FunctionEffects) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Set { name, value } => {
                effects.effects.insert(EffectKind::Write);
                effects.writes.insert(name.clone());
                collect_expr(value, effects);
            }
            StmtKind::SetIndex {
                collection,
                index,
                value,
            } => {
                effects.effects.insert(EffectKind::Write);
                if let ExprKind::Name(name) = &collection.kind {
                    effects.writes.insert(name.clone());
                }
                collect_expr(collection, effects);
                collect_expr(index, effects);
                collect_expr(value, effects);
            }
            StmtKind::Let { name, value } => {
                effects.effects.insert(EffectKind::Write);
                effects.writes.insert(name.clone());
                collect_expr(value, effects);
            }
            StmtKind::Return(value) => {
                if let Some(value) = value {
                    collect_expr(value, effects);
                }
            }
            StmtKind::Expr(expr) => collect_expr(expr, effects),
            StmtKind::Scope { body } => collect_block(body, effects),
            StmtKind::Spawn { call, .. } => {
                effects.effects.insert(EffectKind::Call);
                collect_expr(call, effects);
            }
            StmtKind::While { condition, body } => {
                collect_expr(condition, effects);
                collect_block(body, effects);
            }
            StmtKind::Join { .. } | StmtKind::Cancel { .. } => {}
        }
    }
}

fn collect_expr(expr: &Expr, effects: &mut FunctionEffects) {
    match &expr.kind {
        ExprKind::Name(name) => {
            effects.effects.insert(EffectKind::Read);
            effects.reads.insert(name.clone());
        }
        ExprKind::Binary { left, right, .. } => {
            collect_expr(left, effects);
            collect_expr(right, effects);
        }
        ExprKind::Call { callee, args } => {
            effects.effects.insert(EffectKind::Call);
            if let ExprKind::Name(name) = &callee.kind {
                effects.calls.insert(name.clone());
            } else {
                collect_expr(callee, effects);
            }
            for arg in args {
                collect_expr(arg, effects);
            }
        }
        ExprKind::Group(inner) => collect_expr(inner, effects),
        ExprKind::List(items) => items.iter().for_each(|e| collect_expr(e, effects)),
        ExprKind::Index { collection, index } => {
            collect_expr(collection, effects);
            collect_expr(index, effects);
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_expr(condition, effects);
            collect_block(then_branch, effects);
            if let Some(else_branch) = else_branch {
                collect_block(else_branch, effects);
            }
        }
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::String(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn propagates_callee_effects_to_callers() {
        let module =
            parse("module x\nfn leaf(a: Int) -> Int\n  a\nfn main() -> Int\n  leaf(1)\n").unwrap();
        let model = analyze(&module);
        let main = &model.functions["main"];
        assert!(main.effects.contains(&EffectKind::Call));
        assert!(main.effects.contains(&EffectKind::Read));
        assert!(main.calls.contains("leaf"));
        assert!(model.dependencies["main"].contains("leaf"));
        assert!(model.depends_on("main", "leaf"));
    }

    #[test]
    fn tracks_resource_reads_and_writes_and_invalidates_callers() {
        let module = parse(
            "module x\nfn leaf() -> Int\n  let value = 1\n  value\nfn main() -> Int\n  leaf()\n",
        )
        .unwrap();
        let model = analyze(&module);
        assert!(model.functions["leaf"].writes.contains("value"));
        assert!(model.functions["leaf"].reads.contains("value"));
        assert!(model.invalidated_by_resource("value").contains("leaf"));
        assert!(model.invalidated_by_resource("value").contains("main"));
    }

    #[test]
    fn indexed_assignment_is_a_write_effect() {
        let module = parse(
            "module x\nfn main() -> Int\n  let items = [1]\n  set items[0] = 2\n  items[0]\n",
        )
        .unwrap();
        let model = analyze(&module);
        assert!(model.functions["main"].effects.contains(&EffectKind::Write));
        assert!(model.functions["main"].writes.contains("items"));
    }

    #[test]
    fn records_local_binding_writes() {
        let module = parse("module x\nfn main() -> Int\n  let x = 1\n  x\n").unwrap();
        let model = analyze(&module);
        let main = &model.functions["main"];
        assert!(main.effects.contains(&EffectKind::Write));
        assert!(main.effects.contains(&EffectKind::Read));
    }
}
