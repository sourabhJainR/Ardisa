use std::collections::{BTreeMap, HashMap};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};

use crate::{Block, ExprKind, Item, Module, StmtKind, resource_guard::{ResourceBudget, ResourceLease}, source::Diagnostic};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskTerminal {
    Joined,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSpec {
    pub name: String,
    pub callee: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeReport {
    pub tasks: Vec<TaskSpec>,
}

pub fn analyze(module: &Module) -> Result<Vec<ScopeReport>, Vec<String>> {
    analyze_with_diagnostics(module).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| format!("{}: {}", error.code, error.message))
            .collect()
    })
}

pub fn analyze_with_diagnostics(module: &Module) -> Result<Vec<ScopeReport>, Vec<Diagnostic>> {
    let mut reports = Vec::new();
    let mut errors = Vec::new();
    for item in &module.items {
        let Item::Function(function) = item;
        analyze_block_with_diagnostics(&function.body, &mut reports, &mut errors);
    }
    if errors.is_empty() {
        Ok(reports)
    } else {
        Err(errors)
    }
}

fn analyze_block(block: &Block, reports: &mut Vec<ScopeReport>, errors: &mut Vec<String>) {
    let mut diagnostics = Vec::new();
    analyze_block_with_diagnostics(block, reports, &mut diagnostics);
    errors.extend(diagnostics.into_iter().map(|error| error.message));
}

fn analyze_block_with_diagnostics(
    block: &Block,
    reports: &mut Vec<ScopeReport>,
    errors: &mut Vec<Diagnostic>,
) {
    for stmt in &block.stmts {
        if let StmtKind::Scope { body } = &stmt.kind {
            let mut tasks = HashMap::<String, (TaskState, crate::source::Span)>::new();
            let mut report = ScopeReport { tasks: Vec::new() };
            for child in &body.stmts {
                match &child.kind {
                    StmtKind::Spawn { name, call } => {
                        if tasks
                            .insert(name.clone(), (TaskState::Running, child.span))
                            .is_some()
                        {
                            errors.push(Diagnostic::error(
                                "AIF501",
                                format!("duplicate task '{name}' in scope"),
                                Some(child.span),
                            ));
                            continue;
                        }
                        let callee = match &call.kind {
                            ExprKind::Call { callee, .. } => match &callee.kind {
                                ExprKind::Name(name) => name.clone(),
                                _ => "<dynamic>".into(),
                            },
                            _ => "<invalid>".into(),
                        };
                        report.tasks.push(TaskSpec {
                            name: name.clone(),
                            callee,
                        });
                    }
                    StmtKind::Join { name } | StmtKind::Cancel { name } => {
                        match tasks.get(name).copied() {
                            Some((TaskState::Running, _)) => {
                                let next = if matches!(child.kind, StmtKind::Join { .. }) {
                                    TaskState::Joined
                                } else {
                                    TaskState::Cancelled
                                };
                                tasks.insert(name.clone(), (next, child.span));
                            }
                            Some(_) => errors.push(Diagnostic::error(
                                "AIF504",
                                format!("task '{name}' is already terminal"),
                                Some(child.span),
                            )),
                            None => errors.push(Diagnostic::error(
                                "AIF502",
                                format!("unknown task '{name}' in scope"),
                                Some(child.span),
                            )),
                        }
                    }
                    StmtKind::Scope { .. } => {
                        analyze_block_with_diagnostics(child_block(child), reports, errors)
                    }
                    _ => {}
                }
            }
            for (name, (state, span)) in tasks {
                if state == TaskState::Running {
                    errors.push(Diagnostic::error(
                        "AIF503",
                        format!("task '{name}' must be joined or cancelled before scope exit"),
                        Some(span),
                    ));
                }
            }
            reports.push(report);
            analyze_block_with_diagnostics(body, reports, errors);
        }
    }
}

fn child_block(stmt: &crate::Stmt) -> &Block {
    match &stmt.kind {
        StmtKind::Scope { body } => body,
        _ => unreachable!(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskState {
    Running,
    Joined,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

pub struct TaskHandle<T> {
    token: CancellationToken,
    join: Option<JoinHandle<T>>,
}

impl<T: Send + 'static> TaskHandle<T> {
    pub fn cancel(&self) {
        self.token.cancelled.store(true, Ordering::Release);
    }

    pub fn is_finished(&self) -> bool {
        self.join.as_ref().is_some_and(|handle| handle.is_finished())
    }

    pub fn join(mut self) -> thread::Result<T> {
        self.join.take().expect("task already joined").join()
    }

    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeEvent {
    Spawned(String),
    Joined(String),
    Cancelled(String),
    Failed(String),
}

pub struct StructuredScope {
    tasks: BTreeMap<String, (TaskHandle<()>, TaskState)>,
    leases: BTreeMap<String, ResourceLease>,
    budget: ResourceBudget,
    events: Vec<ScopeEvent>,
}

impl StructuredScope {
    pub fn new() -> Self {
        Self::with_budget(ResourceBudget::default())
    }

    /// Create a scope using a caller-provided shared budget. Cloned budgets share
    /// counters, allowing multiple scopes to obey one process/session limit.
    pub fn with_budget(budget: ResourceBudget) -> Self {
        Self {
            tasks: BTreeMap::new(),
            leases: BTreeMap::new(),
            budget,
            events: Vec::new(),
        }
    }

    pub fn spawn<F>(&mut self, name: impl Into<String>, task: F) -> Result<(), String>
    where
        F: FnOnce(CancellationToken) + Send + 'static,
    {
        let name = name.into();
        if self.tasks.contains_key(&name) {
            return Err(format!("AIF501: duplicate task '{name}'"));
        }
        let lease = self.budget.reserve_task().map_err(|error| format!("AIF506: {error}"))?;
        self.tasks
            .insert(name.clone(), (spawn(task), TaskState::Running));
        self.leases.insert(name.clone(), lease);
        self.events.push(ScopeEvent::Spawned(name));
        Ok(())
    }

    pub fn join(&mut self, name: &str) -> Result<(), String> {
        let Some((handle, state)) = self.tasks.remove(name) else {
            return Err(format!("AIF502: unknown task '{name}'"));
        };
        let lease = self.leases.remove(name);
        // Cancellation is cooperative: a cancelled worker still has to be joined
        // before its resource reservation can be released.
        let result = match handle.join() {
            Ok(()) => {
                if state == TaskState::Running {
                    self.events.push(ScopeEvent::Joined(name.into()));
                }
                Ok(())
            }
            Err(_) => {
                self.events.push(ScopeEvent::Failed(name.into()));
                self.cancel_running_siblings();
                Err(format!(
                    "AIF505: task '{name}' panicked; siblings cancelled"
                ))
            }
        };
        drop(lease);
        result
    }

    pub fn cancel(&mut self, name: &str) -> Result<(), String> {
        let Some((handle, state)) = self.tasks.get_mut(name) else {
            return Err(format!("AIF502: unknown task '{name}'"));
        };
        if *state != TaskState::Running {
            return Err(format!("AIF504: task '{name}' is already terminal"));
        }
        handle.cancel();
        *state = TaskState::Cancelled;
        self.events.push(ScopeEvent::Cancelled(name.into()));
        Ok(())
    }

    pub fn finish(mut self) -> Result<Vec<ScopeEvent>, String> {
        let mut first_failure = None;
        while !self.tasks.is_empty() {
            // Only join completed workers. Joining in name order can otherwise
            // block on a cooperative sibling before discovering a later panic.
            let completed = self.tasks.iter()
                .filter_map(|(name, (handle, _))| handle.is_finished().then_some(name.clone()))
                .collect::<Vec<_>>();
            if completed.is_empty() {
                thread::yield_now();
                continue;
            }
            for name in completed {
                let Some((handle, state)) = self.tasks.remove(&name) else {
                    continue;
                };
                let lease = self.leases.remove(&name);
                let join_result = handle.join();
                drop(lease);
                if join_result.is_err() {
                    self.events.push(ScopeEvent::Failed(name.clone()));
                    if first_failure.is_none() {
                        first_failure = Some(format!("AIF505: task '{name}' panicked"));
                    }
                    self.cancel_running_siblings();
                } else if state == TaskState::Running {
                    self.events.push(ScopeEvent::Joined(name));
                }
            }
        }
        match first_failure {
            Some(error) => Err(error),
            None => Ok(self.events.clone()),
        }
    }

    pub fn events(&self) -> &[ScopeEvent] {
        &self.events
    }

    fn cancel_running_siblings(&mut self) {
        let names = self
            .tasks
            .iter()
            .filter_map(|(name, (_, state))| (*state == TaskState::Running).then_some(name.clone()))
            .collect::<Vec<_>>();
        for name in names {
            if let Some((handle, state)) = self.tasks.get_mut(&name) {
                handle.cancel();
                *state = TaskState::Cancelled;
                self.events.push(ScopeEvent::Cancelled(name));
            }
        }
    }
}

impl Default for StructuredScope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for StructuredScope {
    fn drop(&mut self) {
        for (handle, state) in self.tasks.values_mut() {
            if *state == TaskState::Running {
                handle.cancel();
                *state = TaskState::Cancelled;
            }
        }
        for (_, (handle, _)) in std::mem::take(&mut self.tasks) {
            let _ = handle.join();
        }
        self.leases.clear();
    }
}

pub fn spawn<T, F>(task: F) -> TaskHandle<T>
where
    T: Send + 'static,
    F: FnOnce(CancellationToken) -> T + Send + 'static,
{
    let token = CancellationToken {
        cancelled: Arc::new(AtomicBool::new(false)),
    };
    let child_token = token.clone();
    let join = thread::spawn(move || task(child_token));
    TaskHandle {
        token,
        join: Some(join),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_joined_structured_scope() {
        let module = crate::parse(
            "module x\nfn main()\n  scope\n    spawn worker = work(1)\n    join worker\n",
        )
        .unwrap();
        let reports = analyze(&module).unwrap();
        assert_eq!(reports[0].tasks[0].name, "worker");
    }

    #[test]
    fn rejects_unfinished_task_at_scope_exit() {
        let module =
            crate::parse("module x\nfn main()\n  scope\n    spawn worker = work(1)\n").unwrap();
        let errors = analyze(&module).unwrap_err();
        assert!(errors.iter().any(|e| e.starts_with("AIF503")));
    }

    #[test]
    fn rejects_double_terminal_task_control() {
        let module = crate::parse(
            "module x
fn main()
  scope
    spawn worker = work(1)
    join worker
    cancel worker
",
        )
        .unwrap();
        let errors = analyze(&module).unwrap_err();
        assert!(errors.iter().any(|e| e.starts_with("AIF504")));
    }

    #[test]
    fn structured_scope_cleans_up_unfinished_tasks_on_drop() {
        let mut scope = StructuredScope::new();
        scope
            .spawn("worker", |token| {
                while !token.is_cancelled() {
                    thread::yield_now();
                }
            })
            .unwrap();
        assert_eq!(scope.events(), &[ScopeEvent::Spawned("worker".into())]);
    }

    #[test]
    fn structured_scope_drop_cancels_joins_and_releases_budget_on_early_exit() {
        use crate::resource_guard::{ResourceBudget, ResourceLimits};

        let budget = ResourceBudget::new(ResourceLimits {
            max_tasks: 1,
            max_reserved_bytes: 0,
            max_operations: 0,
        });

        fn return_early(budget: ResourceBudget) {
            let mut scope = StructuredScope::with_budget(budget);
            scope
                .spawn("worker", |token| {
                    while !token.is_cancelled() {
                        thread::yield_now();
                    }
                })
                .unwrap();
            // Returning without finish must still cancel and join the worker via Drop.
        }

        return_early(budget.clone());
        assert_eq!(budget.snapshot().active_tasks, 0);

        // The released lease must make the shared task budget reusable.
        let mut next = StructuredScope::with_budget(budget.clone());
        next.spawn("next", |_token| {}).unwrap();
        next.finish().unwrap();
        assert_eq!(budget.snapshot().active_tasks, 0);
    }

    #[test]
    fn structured_scope_requires_explicit_terminal_state() {
        let mut scope = StructuredScope::new();
        scope.spawn("worker", |_token| {}).unwrap();
        assert!(scope.join("worker").is_ok());
        assert_eq!(
            scope.events(),
            &[
                ScopeEvent::Spawned("worker".into()),
                ScopeEvent::Joined("worker".into())
            ]
        );
    }
    #[test]
    fn structured_scope_cancellation_is_followed_by_scope_cleanup() {
        let mut scope = StructuredScope::new();
        scope
            .spawn("worker", |token| {
                while !token.is_cancelled() {
                    thread::yield_now();
                }
            })
            .unwrap();
        scope.cancel("worker").unwrap();
        let events = scope.finish().unwrap();
        assert_eq!(
            events,
            vec![
                ScopeEvent::Spawned("worker".into()),
                ScopeEvent::Cancelled("worker".into())
            ]
        );
    }

    #[test]
    fn child_failure_cancels_running_siblings() {
        let mut scope = StructuredScope::new();
        scope
            .spawn("a", |_token| {
                panic!("boom");
            })
            .unwrap();
        scope
            .spawn("b", |token| {
                while !token.is_cancelled() {
                    thread::yield_now();
                }
            })
            .unwrap();
        let result = scope.join("a");
        assert!(result.is_err());
        assert!(scope.events().contains(&ScopeEvent::Failed("a".into())));
        assert!(scope.events().contains(&ScopeEvent::Cancelled("b".into())));
    }

    #[test]
    fn scope_event_order_is_deterministic_for_cleanup() {
        let mut scope = StructuredScope::new();
        scope.spawn("z", |_token| {}).unwrap();
        scope.spawn("a", |_token| {}).unwrap();
        let events = scope.finish().unwrap();
        assert_eq!(
            events,
            vec![
                ScopeEvent::Spawned("z".into()),
                ScopeEvent::Spawned("a".into()),
                ScopeEvent::Joined("a".into()),
                ScopeEvent::Joined("z".into()),
            ]
        );
    }

    #[test]
    fn structured_scope_rejects_duplicate_tasks() {
        let mut scope = StructuredScope::new();
        scope.spawn("worker", |_token| {}).unwrap();
        assert!(scope.spawn("worker", |_token| {}).is_err());
    }

    #[test]
    fn cancellation_token_is_cooperative() {
        let handle = spawn(|token| {
            while !token.is_cancelled() {
                thread::yield_now();
            }
            7
        });
        handle.cancel();
        assert_eq!(handle.join().unwrap(), 7);
    }

    #[test]
    fn structured_scope_enforces_shared_task_budget() {
        use crate::resource_guard::{ResourceBudget, ResourceLimits};
        let budget = ResourceBudget::new(ResourceLimits {
            max_tasks: 1,
            max_reserved_bytes: 1024,
            max_operations: 100,
        });
        let mut first = StructuredScope::with_budget(budget.clone());
        first.spawn("worker", |token| {
            while !token.is_cancelled() {
                thread::yield_now();
            }
        }).unwrap();
        let mut second = StructuredScope::with_budget(budget.clone());
        assert!(second.spawn("blocked", |_token| {}).unwrap_err().contains("AIF506"));
        first.cancel("worker").unwrap();
        first.finish().unwrap();
        second.spawn("allowed", |_token| {}).unwrap();
        second.finish().unwrap();
        assert_eq!(budget.snapshot().active_tasks, 0);
    }

    #[test]
    fn cancelled_worker_is_joined_before_budget_is_released() {
        use crate::resource_guard::{ResourceBudget, ResourceLimits};
        let budget = ResourceBudget::new(ResourceLimits { max_tasks: 1, max_reserved_bytes: 0, max_operations: 0 });
        let mut scope = StructuredScope::with_budget(budget.clone());
        scope.spawn("worker", |token| {
            while !token.is_cancelled() { thread::yield_now(); }
        }).unwrap();
        scope.cancel("worker").unwrap();
        assert!(scope.join("worker").is_ok());
        assert_eq!(budget.snapshot().active_tasks, 0);
    }
}
