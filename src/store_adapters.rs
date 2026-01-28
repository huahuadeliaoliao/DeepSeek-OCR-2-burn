use std::rc::Rc;

use burn::store::{ModuleAdapter, TensorSnapshot};
use burn::tensor::{DType, TensorData};

/// Chain two Burn-store adapters.
#[derive(Debug, Clone)]
pub struct ChainAdapter<A, B> {
    pub first: A,
    pub second: B,
}

impl<A, B> ChainAdapter<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A, B> ModuleAdapter for ChainAdapter<A, B>
where
    A: ModuleAdapter + Clone + 'static,
    B: ModuleAdapter + Clone + 'static,
{
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let snapshot = self.first.adapt(snapshot);
        self.second.adapt(&snapshot)
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        // Preserve the lookup behavior (e.g., weight<->gamma for LayerNorm) from the first adapter.
        self.first
            .get_alternative_param_name(param_name, container_type)
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Cast all tensor snapshots to a target dtype during loading.
///
/// This is useful when the backend doesn't support the weight dtype (e.g. BF16 on Burn WebGPU).
#[derive(Debug, Clone)]
pub struct CastDTypeAdapter {
    pub target: DType,
}

impl CastDTypeAdapter {
    pub fn to_f32() -> Self {
        Self { target: DType::F32 }
    }

    pub fn to_f16() -> Self {
        Self { target: DType::F16 }
    }

    fn cast_data(data: TensorData, target: DType) -> TensorData {
        if data.dtype == target {
            data
        } else {
            data.convert_dtype(target)
        }
    }
}

impl ModuleAdapter for CastDTypeAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype == self.target {
            return snapshot.clone();
        }

        let original_fn = snapshot.clone_data_fn();
        let target = self.target;
        let shape = snapshot.shape.clone();

        let path_stack = snapshot.path_stack.clone().unwrap_or_default();
        let container_stack = snapshot.container_stack.clone().unwrap_or_default();
        let tensor_id = snapshot.tensor_id.unwrap_or_default();

        let data_fn = Rc::new(move || {
            let data = original_fn()?;
            Ok(Self::cast_data(data, target))
        });

        TensorSnapshot::from_closure(
            data_fn,
            target,
            shape,
            path_stack,
            container_stack,
            tensor_id,
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Cast tensor snapshots to different dtypes based on their parameter path.
///
/// This is useful when only a subset of the model needs higher precision (e.g. Conv2d in
/// SAM/ViT is unstable on some backends in F16).
#[derive(Debug, Clone)]
pub struct SelectiveCastDTypeAdapter {
    pub default: DType,
    pub rules: Vec<(String, DType)>,
}

impl SelectiveCastDTypeAdapter {
    pub fn new(default: DType) -> Self {
        Self {
            default,
            rules: Vec::new(),
        }
    }

    /// Add a rule that applies when `snapshot.full_path().starts_with(prefix)`.
    pub fn with_prefix(mut self, prefix: impl Into<String>, target: DType) -> Self {
        self.rules.push((prefix.into(), target));
        self
    }

    fn target_for(&self, snapshot: &TensorSnapshot) -> DType {
        let path = snapshot.full_path();
        for (prefix, target) in self.rules.iter() {
            if path.starts_with(prefix) {
                return *target;
            }
        }
        self.default
    }
}

impl ModuleAdapter for SelectiveCastDTypeAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let target = self.target_for(snapshot);
        if snapshot.dtype == target {
            return snapshot.clone();
        }

        let original_fn = snapshot.clone_data_fn();
        let shape = snapshot.shape.clone();

        let path_stack = snapshot.path_stack.clone().unwrap_or_default();
        let container_stack = snapshot.container_stack.clone().unwrap_or_default();
        let tensor_id = snapshot.tensor_id.unwrap_or_default();

        let data_fn = Rc::new(move || {
            let data = original_fn()?;
            Ok(CastDTypeAdapter::cast_data(data, target))
        });

        TensorSnapshot::from_closure(
            data_fn,
            target,
            shape,
            path_stack,
            container_stack,
            tensor_id,
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}
