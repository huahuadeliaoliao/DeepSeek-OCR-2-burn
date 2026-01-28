use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::store::{BurnToPyTorchAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor};
use burn_ndarray::NdArray;

#[derive(burn::module::Module, Debug)]
struct ToyModel<B: Backend> {
    linear: Linear<B>,
    norm: LayerNorm<B>,
}

impl<B: Backend> ToyModel<B> {
    fn new(device: &B::Device) -> Self {
        let linear = LinearConfig::new(4, 3)
            .with_bias(true)
            .with_initializer(burn::module::Initializer::Constant { value: 0.5 })
            .init(device);
        let norm = LayerNormConfig::new(3).init(device);
        Self { linear, norm }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.norm.forward(self.linear.forward(x))
    }
}

#[test]
fn safetensors_pytorch_roundtrip_linear_and_layernorm() {
    type B = NdArray;
    let device = Default::default();

    // 1) Build a model with deterministic params.
    let model = ToyModel::<B>::new(&device);
    let input = Tensor::<B, 2>::ones(Shape::new([2, 4]), &device);
    let expected = model.forward(input.clone()).into_data();

    // 2) Save "PyTorch-style" safetensors (linear weights transposed + LayerNorm gamma/beta renamed).
    let tmp = std::env::temp_dir().join("burn_pytorch_roundtrip.safetensors");
    if tmp.exists() {
        std::fs::remove_file(&tmp).unwrap();
    }
    let mut store = SafetensorsStore::from_file(&tmp)
        .with_to_adapter(BurnToPyTorchAdapter)
        .skip_enum_variants(true);
    model.save_into(&mut store).unwrap();

    // 3) Load back using the PyTorch -> Burn adapter.
    let mut loaded = ToyModel::<B>::new(&device);
    let mut store = SafetensorsStore::from_file(&tmp)
        .with_from_adapter(PyTorchToBurnAdapter)
        .skip_enum_variants(true);
    loaded.load_from(&mut store).unwrap();

    let actual = loaded.forward(input).into_data();
    expected.assert_eq(&actual, true);

    std::fs::remove_file(&tmp).ok();
}
