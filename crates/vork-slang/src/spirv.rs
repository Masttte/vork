/// Outlining bounds Slang's optimization graph; the device compiler still needs
/// freedom to inline large closure returns rather than spill them at each hit.
/// SPIR-V Function Control's DontInline bit is a performance hint, not semantics.
pub fn allow_driver_inlining(code: &mut [u8]) -> anyhow::Result<()> {
    allow_driver_inlining_below(code, usize::MAX)
}

/// Preserve Slang's function boundaries when a module exceeds the caller's
/// instruction budget. Large authored path graphs otherwise multiply again at
/// their material call sites during native driver compilation.
pub fn allow_driver_inlining_below(code: &mut [u8], limit: usize) -> anyhow::Result<()> {
    anyhow::ensure!(code.len() >= 20 && code.len() % 4 == 0, "malformed SPIR-V from Slang");
    anyhow::ensure!(u32::from_le_bytes(code[..4].try_into().unwrap()) == 0x07230203, "invalid SPIR-V magic from Slang");
    let mut offset = 20;
    let mut instructions = 0;
    let mut controls = Vec::new();
    while offset < code.len() {
        let instruction = u32::from_le_bytes(code[offset..offset + 4].try_into().unwrap());
        let words = (instruction >> 16) as usize;
        anyhow::ensure!(words > 0 && words <= (code.len() - offset) / 4, "malformed SPIR-V instruction from Slang");
        if instruction & 0xffff == 54 { // OpFunction: result type, result ID, control, function type.
            anyhow::ensure!(words == 5, "malformed SPIR-V function from Slang");
            controls.push(offset + 12);
        }
        instructions += 1;
        offset += words * 4;
    }
    if instructions <= limit {
        for offset in controls {
            let control = &mut code[offset..offset + 4];
            let flags = u32::from_le_bytes(control.try_into().unwrap());
            control.copy_from_slice(&(flags & !0x2).to_le_bytes());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(tail: &[u32]) -> Vec<u8> {
        [0x07230203, 0x00010500, 0, 3, 0, (5 << 16) | 54, 1, 2, 0xe, 1]
            .into_iter().chain(tail.iter().copied()).flat_map(u32::to_le_bytes).collect()
    }

    #[test]
    fn instruction_budget_preserves_boundaries_and_other_flags() {
        let original = module(&[(1 << 16) | 56]);
        let mut code = original.clone();
        allow_driver_inlining_below(&mut code, 1).unwrap();
        assert_eq!(code, original);
        allow_driver_inlining_below(&mut code, 2).unwrap();
        let mut expected = original.clone();
        expected[32..36].copy_from_slice(&0xcu32.to_le_bytes());
        assert_eq!(code, expected);
        let mut unlimited = original;
        allow_driver_inlining(&mut unlimited).unwrap();
        assert_eq!(unlimited, expected);
    }

    #[test]
    fn over_budget_modules_are_still_validated_before_mutation() {
        let original = module(&[0]);
        let mut code = original.clone();
        assert!(allow_driver_inlining_below(&mut code, 0).is_err());
        assert_eq!(code, original);
        assert!(allow_driver_inlining(&mut code).is_err());
        assert_eq!(code, original);
    }
}
