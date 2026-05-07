pub mod func;

//#[cfg(feature = "sexpr_parser")]
pub mod parser;

//#[cfg(feature = "simplifier")]
pub mod simplifier;

#[cfg(test)]
mod tests {
    use crate::func::EnvFunction;
    #[test]
    fn test_arg() {
        let func = EnvFunction::Arg;
        let vc = (-32000..32000).map(|s| s as f32).collect::<Vec<_>>();
        assert_eq!(vc, vc.iter().map(|&x| func.eval(x)).collect::<Vec<_>>());
    }

    #[test]
    fn test_id() {
        let func = EnvFunction::Id(Box::new(EnvFunction::Arg));
        let vc = (-32000..32000).map(|s| s as f32).collect::<Vec<_>>();
        assert_eq!(vc, vc.iter().map(|&x| func.eval(x)).collect::<Vec<_>>());
    }

    #[test]
    fn test_sine() {
        let func = EnvFunction::Sin(Box::new(EnvFunction::Arg));
        (-32000..32000)
            .map(|s| s as f32)
            .for_each(|x| assert!((x.sin() - func.eval(x)).abs() < 1e-3));
    }
}
