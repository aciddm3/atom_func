use crate::func::EnvFunction;

impl EnvFunction {
    #[inline]
    pub fn simplify<const MAX_RECURSION_DEPTH: usize>(&mut self) {
        self.simplify_with_depth::<MAX_RECURSION_DEPTH>(0);
    }

    fn simplify_with_depth<const MAX_RECURSION_DEPTH: usize>(&mut self, depth: usize) {
        if depth > MAX_RECURSION_DEPTH {
            return;
        } //  stack overflow protection

        match self {
            Self::Id(inner)
            | Self::Exp(inner)
            | Self::Sigmoid(inner)
            | Self::Ln(inner)
            | Self::Sin(inner)
            | Self::Cos(inner)
            | Self::InvVal(inner)
            | Self::Neg(inner)
            | Self::Abs(inner)
            | Self::Linear(inner, _, _)
            | Self::Powf(inner, _)
            | Self::Powi(inner, _)
            | Self::Periodic(inner, _)
            | Self::ConstMul(inner, _) => {
                inner.simplify_with_depth::<MAX_RECURSION_DEPTH>(depth + 1);
            }
            Self::Sum(a, b) | Self::Dif(a, b) | Self::Mul(a, b) | Self::Div(a, b) => {
                a.simplify_with_depth::<MAX_RECURSION_DEPTH>(depth + 1);
                b.simplify_with_depth::<MAX_RECURSION_DEPTH>(depth + 1);
            }
            _ => {}
        }
        self.apply_simp_rules();
    }

    #[inline]
    fn apply_simp_rules(&mut self) {
        match self {
            // ─────────────────────────────────────────────────────
            // 🔹 Базовые тождества (убираем "шум" в дереве)
            // ─────────────────────────────────────────────────────
            Self::Id(inner) => {
                // Правило: `Id(x) → x`
                // Почему: `Id` — это техническая обёртка, не несёт семантики.
                // Убираем её, чтобы не делать лишний вызов функции в eval().
                *self = *inner.clone();
            }

            // ─────────────────────────────────────────────────────
            // 🔹 Умножение на константу (частый кейс: масштабирование огибающей)
            // ─────────────────────────────────────────────────────
            Self::ConstMul(inner, c) => {
                // Правило: `0 * f(x) → 0`
                // Почему: если коэффициент 0, вся ветка обнуляется.
                // Это позволяет "выключить" модуляцию без удаления узла из дерева.
                if *c == 0.0 {
                    *self = Self::Constant(0.0);
                    return;
                }
                // Правило: `1 * f(x) → f(x)`
                // Почему: умножение на 1 — это операция-пустышка. Убираем её,
                // чтобы `eval()` не тратил такты на умножение и рекурсию.
                if *c == 1.0 {
                    *self = *inner.clone();
                    return;
                }
                // Правило: `c * Constant(v) → Constant(c*v)`
                // Почему: сворачиваем константы заранее. В audio-потоке это
                // превращает 2 операции (умножение + чтение константы) в 1 чтение.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(*c * *v);
                }
                // Правило: `a * (b * x) → (a*b) * x`
                // Почему: ассоциативность умножения. Схлопываем цепочки `ConstMul`,
                // чтобы дерево было плоским, а не глубоким. Меньше глубины =
                // меньше рекурсии в eval() + лучше кэширование.
                else if let Self::ConstMul(inner2, b) = inner.as_mut() {
                    *self = Self::ConstMul(inner2.clone(), *b * *c);
                }
            }

            // ─────────────────────────────────────────────────────
            // 🔹 Унарные функции: сворачивание констант (пре-вычисление)
            // ─────────────────────────────────────────────────────
            Self::Exp(inner) => {
                // Правило: `exp(Constant(v)) → Constant(exp(v))`
                // Почему: `exp()` — дорогая функция (~20-50 тактов). Если аргумент
                // константа, вычисляем её один раз при загрузке пресета, а не
                // на каждый сэмпл в audio-потоке.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(v.exp());
                }
            }

            Self::Sigmoid(inner) => {
                // Правило: `sigmoid(Constant(v)) → Constant(sigmoid(v))`
                // Почему: сигмоида используется для "мягкого клиппинга" огибающей.
                // Если вход константа, пре-вычисляем, чтобы в real-time не считать
                // `1.0 / (1.0 + exp(-v))` на каждом сэмпле.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(1.0 / (1.0 + (-v).exp()));
                }
            }

            Self::Ln(inner) => {
                // Правило: `ln(Constant(v)) → Constant(ln(v))`
                // Почему: логарифм нужен для экспоненциальных кривых (например,
                // частотная модуляция в лог-шкале). Пре-вычисление экономит ~30 тактов/сэмпл.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(v.ln());
                }
            }

            Self::Sin(inner) => {
                // Правило: `sin(Constant(v)) → Constant(sin(v))`
                // Почему: синус — основа для LFO и вибрато. Если фаза фиксирована
                // (константа), нет смысла вычислять тригонометрию в audio-потоке.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(v.sin());
                }
            }

            Self::Cos(inner) => {
                // Правило: `cos(Constant(v)) → Constant(cos(v))`
                // Почему: аналогично синусу. Косинус часто используется для
                // фазового сдвига на π/2 относительно синуса.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(v.cos());
                }
                // Правило: `cos(|x|) → cos(x)` и `cos(-x) → cos(x)`
                // Почему: косинус — чётная функция. Убираем лишние `Abs`/`Neg`,
                // чтобы дерево было компактнее, а `eval()` быстрее.
                else if let Self::Abs(inner2) = inner.as_mut() {
                    *inner = inner2.clone();
                } else if let Self::Neg(inner2) = inner.as_mut() {
                    *inner = inner2.clone();
                }
            }

            // ─────────────────────────────────────────────────────
            // 🔹 Линейные преобразования (основа модуляции)
            // ─────────────────────────────────────────────────────
            Self::Linear(inner, k, b) => {
                // Правило: `0*x + b → Constant(b)`
                // Почему: если коэффициент при x равен 0, функция вырождается
                // в константу. Это позволяет "заморозить" модуляцию на фиксированном уровне.
                if *k == 0.0 {
                    *self = Self::Constant(*b);
                    return;
                }
                // Правило: `k*x + 0 → ConstMul(x, k)`
                // Почему: если свободный член 0, `Linear` избыточен. Заменяем на
                // более простой `ConstMul`, у которого `eval()` на 1 операцию быстрее.
                if *b == 0.0 {
                    *self = Self::ConstMul(inner.clone(), *k);
                    return;
                }
                // Правило: `k*Constant(v) + b → Constant(k*v + b)`
                // Почему: сворачиваем аффинное преобразование константы.
                // В audio-потоке это 1 чтение вместо 3 операций (умножение + сложение + чтение).
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(*k * *v + *b);
                }
            }

            // ─────────────────────────────────────────────────────
            // 🔹 Степени (для нелинейных огибающих, дисторшна)
            // ─────────────────────────────────────────────────────
            Self::Powf(inner, p) => {
                if *p == 0.0 {
                    *self = Self::Constant(1.0);
                    return;
                }
                if *p == 1.0 {
                    *self = *inner.clone();
                    return;
                }
                // Правило: `Constant(v)^p → Constant(v^p)`
                // Почему: возведение в степень — дорогая операция (~40 тактов).
                // Пре-вычисление для констант критично для производительности.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(v.powf(*p));
                } else
                // Правило: `Pow(Pow(x, p2), p1) = Pow(x, p1 * p2)`
                // Почему: возведение в степень — дорогая операция (~40 тактов).
                // Выгоднее потрать такты процессора на возведение и умножение,
                // чем на два возведения
                if let Self::Powf(v, p2) = inner.as_mut() {
                    *self = Self::Powf(v.clone(), *p * *p2);
                } else if let Self::Powi(v, p2) = inner.as_mut() {
                    *self = Self::Powf(v.clone(), *p * *p2 as f32);
                }
            }

            Self::Powi(inner, p) => {
                // Правило: `Constant(v)^n → Constant(v^n)`
                // Почему: целочисленная степень быстрее дробной, но всё равно
                // лучше вычислить один раз при загрузке, чем на каждый сэмпл.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(v.powi(*p));
                } else if let Self::Powf(v, p2) = inner.as_ref() {
                    *self = Self::Powf(v.clone(), *p2 * *p as f32);
                } else if let Self::Powi(v, p2) = inner.as_ref() {
                    *self = Self::Powi(v.clone(), *p2 * *p);
                } else {
                    // Правило: `x^0 → 1`
                    // Почему: математическое тождество. Позволяет "выключить" нелинейность,
                    // не удаляя узел из дерева (удобно для автоматизации параметров).
                    if *p == 0 {
                        *self = Self::Constant(1.0);
                        return;
                    }
                    // Правило: `x^1 → x`
                    // Почему: степень 1 — это тождественное преобразование. Убираем
                    // лишний вызов `powi()` в eval().
                    if *p == 1 {
                        *self = *inner.clone();
                        return;
                    }
                    // Правило: `(|x|)^(2n) → x^(2n)` и `(-x)^(2n) → x^(2n)`
                    // Почему: чётная степень "съедает" знак. Убираем лишние `Abs`/`Neg`,
                    // чтобы упростить дерево. Важно для дисторшна: `x^2` уже всегда ≥ 0.
                    if *p % 2 == 0 {
                        if let Self::Abs(inner2) = inner.as_mut() {
                            *inner = inner2.clone();
                        }
                        if let Self::Neg(inner2) = inner.as_mut() {
                            *inner = inner2.clone();
                        }
                    }
                }
            }

            // ─────────────────────────────────────────────────────
            // 🔹 Периодичность (для LFO, циклических огибающих)
            // ─────────────────────────────────────────────────────
            Self::Periodic(inner, _) => {
                // Правило: `Periodic(Constant(v), p) → Constant(v)`
                // Почему: если функция не зависит от аргумента (константа),
                // периодичность бессмысленна. Убираем обёрку, чтобы не делать
                // `rem_euclid()` на каждый сэмпл впустую.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(*v);
                }
            }

            // ─────────────────────────────────────────────────────
            // 🔹 Обратные функции (для деления, нормализации)
            // ─────────────────────────────────────────────────────
            Self::InvVal(inner) => {
                const EPS: f32 = 1e-7;
                // Правило: `1/Constant(v) → Constant(1/v)`
                // Почему: деление — одна из самых дорогих операций (~15-30 тактов).
                // Пре-вычисление для констант обязательно. Защита от деления на 0
                // предотвращает `Inf`, который вызовет щелчок в аудио.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = if v.abs() < EPS {
                        Self::Constant(0.0)
                    } else {
                        Self::Constant(1.0 / *v)
                    };
                }
                // Правило: `1/(1/x) → x`
                // Почему: двойная инверсия — это тождество. Убираем два вызова
                // деления в eval(), оставляя просто чтение аргумента.
                else if let Self::InvVal(inner2) = inner.as_mut() {
                    *self = *inner2.clone();
                } else if let Self::Powf(inner2, p) = inner.as_ref() {
                    *self = Self::Powf(inner2.clone(), -*p);
                } else if let Self::Powi(inner2, p) = inner.as_ref() {
                    *self = Self::Powi(inner2.clone(), -*p);
                }
            }

            // ─────────────────────────────────────────────────────
            // 🔹 Знаковые операции (для выпрямления, симметрии)
            // ─────────────────────────────────────────────────────
            Self::Neg(inner) => {
                // Правило: `-Constant(v) → Constant(-v)`
                // Почему: унарный минус на константе сворачивается в одну операцию.
                // В eval() это избавляет от лишнего вычитания.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(-*v);
                }
                // Правило: `-(-x) → x`
                // Почему: двойное отрицание — тождество. Убираем два вызова
                // унарного минуса в eval(), что особенно важно при глубокой модуляции.
                else if let Self::Neg(inner2) = inner.as_mut() {
                    *self = *inner2.clone();
                }
            }

            Self::Abs(inner) => {
                // Правило: `|Constant(v)| → Constant(|v|)`
                // Почему: модуль константы вычисляется один раз. В eval() не нужно
                // делать ветвление или вычитание для знака.
                if let Self::Constant(v) = inner.as_ref() {
                    *self = Self::Constant(v.abs());
                }
                // Правило: `||x|| → |x|`
                // Почему: модуль от модуля избыточен. Убираем лишний уровень дерева.
                else if let Self::Abs(inner2) = inner.as_mut() {
                    *inner = inner2.clone();
                }
                // Правило: `|-x| → |x|`
                // Почему: модуль "съедает" знак. Это позволяет упростить выражения
                // вида `Abs(Neg(Sin(t)))` до `Abs(Sin(t))`, что на 1 рекурсию меньше в eval().
                else if let Self::Neg(inner2) = inner.as_mut() {
                    *inner = inner2.clone();
                }
            }

            // ─────────────────────────────────────────────────────
            // 🔹 Бинарные операции: арифметика констант и линейность
            // ─────────────────────────────────────────────────────
            Self::Sum(l, r) => {
                // Правило: `Constant(a) + Constant(b) → Constant(a+b)`
                // Почему: сложение двух констант — кандидат на пре-вычисление.
                // В eval() это превращает 3 операции (2 чтения + сложение) в 1 чтение.
                if let (Self::Constant(a), Self::Constant(b)) = (l.as_ref(), r.as_ref()) {
                    *self = Self::Constant(*a + *b);
                    return;
                }
                // Правило: `Constant(c) + f(x) → Linear(f, 1, c)`
                // Почему: константа слева — это сдвиг по оси Y. Приводим к `Linear`,
                // чтобы в eval() делать `k*val + b` за 2 операции вместо рекурсии в два поддерева.
                if let Self::Constant(v) = l.as_ref() {
                    *self = Self::Linear(r.clone(), 1.0, *v);
                    return;
                }
                // Правило: `f(x) + Constant(c) → Linear(f, 1, c)`
                // Почему: аналогично, но константа справа. Коммутативность сложения
                // позволяет унифицировать оба случая в `Linear`.
                if let Self::Constant(v) = r.as_ref() {
                    *self = Self::Linear(l.clone(), 1.0, *v);
                }
            }

            Self::Dif(l, r) => {
                // Правило: `Constant(a) - Constant(b) → Constant(a-b)`
                // Почему: вычитание констант сворачивается заранее. В eval() экономим
                // на рекурсии и одной операции вычитания.
                if let (Self::Constant(a), Self::Constant(b)) = (l.as_ref(), r.as_ref()) {
                    *self = Self::Constant(*a - *b);
                    return;
                }
                // Правило: `Constant(c) - f(x) → Linear(f, -1, c)`
                // Почему: превращаем вычитание в аффинное преобразование: `c - f = (-1)*f + c`.
                // Это позволяет eval() использовать быстрый путь `Linear` вместо рекурсии.
                if let Self::Constant(v) = l.as_ref() {
                    *self = Self::Linear(r.clone(), -1.0, *v);
                    return;
                }
                // Правило: `f(x) - Constant(c) → Linear(f, 1, -c)`
                // Почему: аналогично: `f - c = 1*f + (-c)`. Унифицируем с `Linear`
                // для более быстрого вычисления в audio-потоке.
                if let Self::Constant(v) = r.as_ref() {
                    *self = Self::Linear(l.clone(), 1.0, -*v);
                }
            }

            Self::Mul(l, r) => {
                // Правило: `Constant(a) * Constant(b) → Constant(a*b)`
                // Почему: умножение констант — кандидат на пре-вычисление.
                // В eval() это 1 чтение вместо 3 операций (2 чтения + умножение).
                if let (Self::Constant(a), Self::Constant(b)) = (l.as_ref(), r.as_ref()) {
                    *self = Self::Constant(*a * *b);
                    return;
                }
                // Правило: `Constant(c) * f(x) → ConstMul(f, c)`
                // Почему: выносим константу в специализированный узел `ConstMul`,
                // у которого eval() оптимизирован под одно умножение (без рекурсии в левое поддерево).
                if let Self::Constant(c) = l.as_ref() {
                    *self = Self::ConstMul(r.clone(), *c);
                    return;
                }
                // Правило: `f(x) * Constant(c) → ConstMul(f, c)`
                // Почему: коммутативность умножения. Унифицируем оба порядка
                // в `ConstMul` для более быстрого eval().
                if let Self::Constant(c) = r.as_ref() {
                    *self = Self::ConstMul(l.clone(), *c);
                }
            }

            Self::Div(l, r) => {
                const EPS: f32 = 1e-7;
                // Правило: `Constant(a) / Constant(b) → Constant(a/b)`
                // Почему: деление констант вычисляем заранее. Защита от деления на 0
                // предотвращает `Inf`, который в audio-потоке = щелчок или краш.
                if let (Self::Constant(a), Self::Constant(b)) = (l.as_ref(), r.as_ref()) {
                    *self = if b.abs() < EPS {
                        Self::Constant(0.0)
                    } else {
                        Self::Constant(*a / *b)
                    };
                    return;
                }
                // Правило: `f(x) / Constant(c) → ConstMul(f, 1/c)`
                // Почему: деление на константу заменяем на умножение на обратную величину.
                // Умножение (~3-5 тактов) быстрее деления (~15-30 тактов) на большинстве CPU.
                // Это критично при 44.1kHz × N голосов.
                if let Self::Constant(c) = r.as_ref() {
                    if c.abs() < EPS {
                        *self = Self::Constant(0.0);
                    } else {
                        *self = Self::ConstMul(l.clone(), 1.0 / *c);
                    }
                    return;
                }
            }
            // `Constant`, `Arg` и неизвестные варианты не упрощаются.
            // Это терминирующие случаи рекурсии.
            _ => {}
        }
    }
}

#[cfg(test)]
mod test {
    use crate::func::EnvFunction;
    fn assert_eval_eq(f: &EnvFunction, x: f32, expected: f32, tolerance: f32) {
        let actual = f.eval(x);
        assert!(
            (actual - expected).abs() < tolerance,
            "f({}) = {}, expected {}",
            x,
            actual,
            expected
        );
    }

    #[test]
    fn test_simplify_id_elimination() {
        let mut f = EnvFunction::Id(Box::new(EnvFunction::Constant(42.0)));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 42.0, 1e-5);
    }

    #[test]
    fn test_simplify_constant_folding_unary() {
        // Exp(Constant(0)) → Constant(1)
        let mut f = EnvFunction::Exp(Box::new(EnvFunction::Constant(0.0)));
        f.simplify::<128>();
        assert_eval_eq(&f, 123.45, 1.0, 1e-5); // x не важен, функция константна

        // Sin(Constant(π/2)) ≈ 1
        let mut f = EnvFunction::Sin(Box::new(EnvFunction::Constant(std::f32::consts::FRAC_PI_2)));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 1.0, 1e-4);
    }

    #[test]
    fn test_simplify_const_mul() {
        // 0 * f(x) → 0
        let mut f =
            EnvFunction::ConstMul(Box::new(EnvFunction::Sin(Box::new(EnvFunction::Arg))), 0.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 1.0, 0.0, 1e-5);

        // 1 * f(x) → f(x)
        let mut f = EnvFunction::ConstMul(Box::new(EnvFunction::Arg), 1.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 5.0, 5.0, 1e-5);

        // c * Constant(v) → Constant(c*v)
        let mut f = EnvFunction::ConstMul(Box::new(EnvFunction::Constant(3.0)), 4.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 12.0, 1e-5);
    }

    #[test]
    fn test_simplify_linear() {
        // k=0: Linear(f, 0, b) → Constant(b)
        let mut f = EnvFunction::Linear(Box::new(EnvFunction::Arg), 0.0, 7.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 100.0, 7.0, 1e-5);

        // b=0: Linear(f, k, 0) → ConstMul(f, k)
        let mut f = EnvFunction::Linear(Box::new(EnvFunction::Arg), 2.0, 0.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 3.0, 6.0, 1e-5);
    }

    #[test]
    fn test_simplify_double_neg() {
        // Neg(Neg(x)) → x
        let mut f = EnvFunction::Neg(Box::new(EnvFunction::Neg(Box::new(EnvFunction::Arg))));
        f.simplify::<128>();
        assert_eval_eq(&f, -5.0, -5.0, 1e-5);
        assert_eval_eq(&f, 3.14, 3.14, 1e-5);
    }

    #[test]
    fn test_simplify_abs_abs() {
        // Abs(Abs(x)) → Abs(x)
        let mut f = EnvFunction::Abs(Box::new(EnvFunction::Abs(Box::new(EnvFunction::Arg))));
        f.simplify::<128>();
        assert_eval_eq(&f, -7.0, 7.0, 1e-5);
        assert_eval_eq(&f, 2.0, 2.0, 1e-5);
    }

    #[test]
    fn test_simplify_abs_neg() {
        // Abs(Neg(x)) → Abs(x)
        let mut f = EnvFunction::Abs(Box::new(EnvFunction::Neg(Box::new(EnvFunction::Arg))));
        f.simplify::<128>();
        assert_eval_eq(&f, -4.0, 4.0, 1e-5);
    }

    #[test]
    fn test_simplify_inv_inv() {
        // InvVal(InvVal(x)) → x (для x ≠ 0)
        let mut f = EnvFunction::InvVal(Box::new(EnvFunction::InvVal(Box::new(EnvFunction::Arg))));
        f.simplify::<128>();
        assert_eval_eq(&f, 2.0, 2.0, 1e-5);
        assert_eval_eq(&f, -3.0, -3.0, 1e-5);
        // Для 0: защита от деления на ноль → 0
        assert_eval_eq(&f, 0.0, 0.0, 1e-5);
    }

    // ─────────────────────────────────────────────────────
    // ➕ Суммы и разности с константами
    // ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_sum_constant_left() {
        // Constant(c) + f(x) → Linear(f, 1, c)
        let mut f = EnvFunction::Sum(
            Box::new(EnvFunction::Constant(5.0)),
            Box::new(EnvFunction::Arg),
        );
        f.simplify::<128>();
        assert_eval_eq(&f, 10.0, 15.0, 1e-5); // 10 + 5
    }

    #[test]
    fn test_simplify_sum_constant_right() {
        // f(x) + Constant(c) → Linear(f, 1, c)
        let mut f = EnvFunction::Sum(
            Box::new(EnvFunction::Arg),
            Box::new(EnvFunction::Constant(-3.0)),
        );
        f.simplify::<128>();
        assert_eval_eq(&f, 10.0, 7.0, 1e-5); // 10 - 3
    }

    #[test]
    fn test_simplify_dif_constant() {
        // Constant(c) - f(x) → Linear(f, -1, c)
        let mut f = EnvFunction::Dif(
            Box::new(EnvFunction::Constant(10.0)),
            Box::new(EnvFunction::Arg),
        );
        f.simplify::<128>();
        assert_eval_eq(&f, 4.0, 6.0, 1e-5); // 10 - 4

        // f(x) - Constant(c) → Linear(f, 1, -c)
        let mut f2 = EnvFunction::Dif(
            Box::new(EnvFunction::Arg),
            Box::new(EnvFunction::Constant(2.0)),
        );
        f2.simplify::<128>();
        assert_eval_eq(&f2, 5.0, 3.0, 1e-5); // 5 - 2
    }

    #[test]
    fn test_simplify_periodic_constant() {
        // Periodic(Constant(v), p) → Constant(v)
        let mut f = EnvFunction::Periodic(Box::new(EnvFunction::Constant(1.5)), 2.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 100.0, 1.5, 1e-5);
    }

    #[test]
    fn test_simplify_nested_simplify() {
        // Exp(Sin(Constant(0))) → Exp(0) → 1
        let mut f = EnvFunction::Exp(Box::new(EnvFunction::Sin(Box::new(EnvFunction::Constant(
            0.0,
        )))));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 1.0, 1e-5);
    }

    // ─────────────────────────────────────────────────────
    // 🛡 Защита от переполнения стека
    // ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_depth_limit() {
        // Создаём дерево глубины 50: Id(Id(Id(...(Arg)...)))
        let mut deep = EnvFunction::Arg;
        for _ in 0..50 {
            deep = EnvFunction::Id(Box::new(deep));
        }

        // Не должно паниковать — защита на 32 уровнях
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            deep.simplify::<128>();
        }));

        assert!(result.is_ok(), "simplify паникнул на глубоком дереве");
    }

    #[test]
    fn test_simplify_depth_limit_nested_ops() {
        // Глубокое дерево с операциями: Exp(Exp(Exp(...)))
        let mut deep = EnvFunction::Arg;
        for _ in 0..40 {
            deep = EnvFunction::Exp(Box::new(deep));
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            deep.simplify::<128>();
        }));

        assert!(result.is_ok(), "simplify паникнул на глубоком Exp-дереве");
    }

    // ─────────────────────────────────────────────────────
    // 🔒 Безопасность eval после simplify
    // ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_nan_safety() {
        // Ln(Constant(-1)) → NaN → 0 (защита в eval)
        let mut f = EnvFunction::Ln(Box::new(EnvFunction::Constant(-1.0)));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 0.0, 1e-5); // NaN превращается в 0
    }

    #[test]
    fn test_simplify_div_zero_safety() {
        // InvVal(Constant(0)) → 0 (защита в eval)
        let mut f = EnvFunction::InvVal(Box::new(EnvFunction::Constant(0.0)));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 0.0, 1e-5);
    }

    // ─────────────────────────────────────────────────────
    // 🎯 Интеграционный тест: сложное выражение
    // ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_complex_expression() {
        // f(x) = 2 * (Sin(x) + 3) - 1
        // После simplify: Linear(ConstMul(Sin(Arg), 2), 1, 5) или эквивалент
        let f = EnvFunction::Dif(
            Box::new(EnvFunction::ConstMul(
                Box::new(EnvFunction::Sum(
                    Box::new(EnvFunction::Sin(Box::new(EnvFunction::Arg))),
                    Box::new(EnvFunction::Constant(3.0)),
                )),
                2.0,
            )),
            Box::new(EnvFunction::Constant(1.0)),
        );

        let mut f_simplified = f.clone();
        f_simplified.simplify::<128>();

        // Проверяем, что значения совпадают на нескольких точках
        let points = vec![0.0, 1.0, std::f32::consts::PI / 2.0, std::f32::consts::PI];
        for x in points {
            let original = f.eval(x);
            let simplified = f_simplified.eval(x);
            assert!(
                (original - simplified).abs() < 1e-4,
                "Расхождение при x={}: {} vs {}",
                x,
                original,
                simplified
            );
        }
    }
    #[test]
    fn test_pow_composition() {
        // (x^2)^3 → x^6
        let mut f = EnvFunction::Powf(
            Box::new(EnvFunction::Powi(Box::new(EnvFunction::Arg), 2)),
            3.0,
        );
        f.simplify::<64>();

        if let EnvFunction::Powf(inner, p) = &f {
            assert!(matches!(**inner, EnvFunction::Arg));
            assert!((p - 6.0).abs() < 1e-5);
        } else {
            panic!("Expected Powf(Arg, 6.0)");
        }
    }

    #[test]
    fn test_invval_pow() {
        // 1/(x^2) → x^(-2)
        let mut f = EnvFunction::InvVal(Box::new(EnvFunction::Powi(Box::new(EnvFunction::Arg), 2)));
        f.simplify::<64>();

        if let EnvFunction::Powi(inner, p) = &f {
            assert!(matches!(**inner, EnvFunction::Arg));
            assert_eq!(*p, -2);
        } else {
            panic!("Expected Powi(Arg, -2)");
        }
    }
}
