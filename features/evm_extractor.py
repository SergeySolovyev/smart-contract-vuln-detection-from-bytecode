from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from pyevmasm import disassemble_all
import multiprocessing as mp
import numpy as np
from scipy.stats import entropy
from collections import Counter
import pandas as pd

class EVMBytecodeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Трансформер признаков из EVM-байткода для задач детекции уязвимостей смарт-контрактов.
    """
    def __init__(self, bytecode_column="bytecode", n_workers=None):
        self.bytecode_column = bytecode_column
        self.n_workers = n_workers
        # Фиксированный список всех выходных признаков
        self.feature_names_ = [
            # Базовые
            "total_instructions", "unique_instructions",
            # Block dependence
            "block_dependent_count", "block_dependency_index",
            "has_TIMESTAMP", "has_NUMBER", "has_DIFFICULTY", "has_GASLIMIT",
            "has_COINBASE", "has_BLOCKHASH",
            # Environmental
            "environmental_instructions_count", "environmental_ratio",
            "unique_environmental_ops", "environmental_complexity",
            # Specific ops
            "balance_operations", "address_operations", "caller_operations",
            "origin_operations", "callvalue_operations", "external_dependency_index",
            # Calldata
            "calldata_size_ops", "calldata_load_ops", "calldata_copy_ops",
            "total_calldata_ops", "calldata_density",
            # External calls
            "external_call_count", "has_external_calls", "call_value_ops",
            "call_gas_limit_ops", "potential_reentrancy_pattern",
            # Memory
            "reads_from_memory", "writes_to_memory", "memory_access_ratio",
            # Stack
            "pushes", "pops", "stack_imbalance", "stack_operations_ratio",
            "stack_underflow_risk",
            # Gas
            "total_gas_cost", "avg_gas_per_instruction", "max_gas_instruction",
            "high_gas_instructions", "gas_dos_risk_index",
            # Arithmetic
            "arithmetic_ops_count", "arithmetic_density", "unsafe_arithmetic_pattern",
            # Control flow
            "control_flow_ops", "jumpi_count", "conditional_branching_ratio",
            "control_flow_complexity",
            # Access control
            "caller_based_checks", "origin_usage", "access_control_ratio",
            "uses_origin_instead_caller",
            # Advanced patterns
            "balance_before_external_call", "randomness_ops_count",
            "has_bad_randomness_pattern",
            # Complexity
            "dangerous_ops_count", "dangerous_ops_density", "opcode_entropy",
            # Composite scores
            "reentrancy_risk_score", "frontrunning_risk_score",
            "dos_risk_score", "arithmetic_risk_score", "overall_security_risk_score",
            # Binary flags
            "has_reentrancy_indicators", "has_unchecked_external_calls",
            "has_arithmetic_vulnerabilities", "has_access_control_issues",
            "has_dos_vulnerabilities",
        ]

    def _extract_features_single(self, bytecode) -> dict:
        """Извлечение признаков из одного байткода (hex-строка или bytes)"""
        if isinstance(bytecode, str):
            bytecode = bytecode.strip()
            if bytecode.startswith("0x"):
                bytecode = bytecode[2:]
            if not bytecode:
                bytecode_bytes = b""
            else:
                try:
                    bytecode_bytes = bytes.fromhex(bytecode)
                except ValueError:
                    bytecode_bytes = b""
        else:
            bytecode_bytes = bytes(bytecode) if bytecode is not None else b""

        try:
            instructions = list(disassemble_all(bytecode_bytes))
        except Exception:
            instructions = []
        n = len(instructions)
        if n == 0:
            return {name: 0.0 for name in self.feature_names_}

        # Частотный счётчик мнемоник
        mnemonic_counter = Counter(instr.mnemonic for instr in instructions)

        # Списки для быстрых проверок
        block_dependent = {"TIMESTAMP", "NUMBER", "DIFFICULTY", "GASLIMIT", "COINBASE", "BLOCKHASH"}
        call_ops = {"CALL", "DELEGATECALL", "STATICCALL", "CALLCODE"}
        arithmetic_ops = {"ADD", "SUB", "MUL", "DIV", "MOD", "SDIV", "SMOD", "EXP", "SIGNEXTEND"}
        dangerous_ops = block_dependent.union(call_ops).union(arithmetic_ops).union({"SELFDESTRUCT"})
        randomness_ops = {"BLOCKHASH", "TIMESTAMP", "DIFFICULTY", "COINBASE"}

        # Базовые агрегаты
        total_instructions = n
        unique_instructions = len(mnemonic_counter)

        block_dependent_count = sum(mnemonic_counter.get(op, 0) for op in block_dependent)

        environmental_count = sum(1 for instr in instructions if getattr(instr, "group", None) == "Environmental Information")
        unique_env_ops = len({instr.mnemonic for instr in instructions if getattr(instr, "group", None) == "Environmental Information"})

        balance_ops = mnemonic_counter.get("BALANCE", 0)
        caller_ops = mnemonic_counter.get("CALLER", 0)
        origin_ops = mnemonic_counter.get("ORIGIN", 0)
        callvalue_ops = mnemonic_counter.get("CALLVALUE", 0)

        calldata_size = mnemonic_counter.get("CALLDATASIZE", 0)
        calldata_load = mnemonic_counter.get("CALLDATALOAD", 0)
        calldata_copy = mnemonic_counter.get("CALLDATACOPY", 0)

        external_call_count = sum(mnemonic_counter.get(op, 0) for op in call_ops)
        gas_ops = mnemonic_counter.get("GAS", 0)  # часто перед CALL

        # Агрегаты по атрибутам инструкций
        pushes = sum(getattr(instr, "pushes", 0) for instr in instructions)
        pops = sum(getattr(instr, "pops", 0) for instr in instructions)
        fees = [getattr(instr, "fee", 0) for instr in instructions]

        total_gas = sum(fees)
        avg_gas = total_gas / n if n else 0
        max_gas = max(fees) if fees else 0
        high_gas_count = sum(f > 1000 for f in fees)

        # Простые паттерны на основе PC (с оговоркой: не идеально из-за JUMP, но как эвристика)
        arithmetic_ops_set = {"ADD", "SUB", "MUL", "DIV", "MOD", "SDIV", "SMOD", "EXP", "SIGNEXTEND"}
        target_mnemonics = {"SSTORE", "CALL", "DELEGATECALL", "STATICCALL", "CALLCODE", "BALANCE", "JUMPI"} | arithmetic_ops_set

        # Только те, что есть в контракте
        target_mnemonics = target_mnemonics & mnemonic_counter.keys()

        pcs = {
            mnemonic: sorted(instr.pc for instr in instructions if instr.mnemonic == mnemonic)
            for mnemonic in target_mnemonics
        }

        potential_reentrancy = 0
        if "SSTORE" in pcs and any(op in pcs for op in call_ops):
            for s_pc in pcs["SSTORE"]:
                for c_op in call_ops:
                    for c_pc in pcs.get(c_op, []):
                        if c_pc > s_pc and (c_pc - s_pc) < 20:
                            potential_reentrancy = 1
                            break

        unsafe_arith = 0
        jumpi_pcs = pcs.get("JUMPI", [])
        for op in arithmetic_ops:
            for a_pc in pcs.get(op, []):
                for j_pc in jumpi_pcs:
                    if j_pc > a_pc and (j_pc - a_pc) < 5:
                        unsafe_arith = 1
                        break

        balance_before_call = 0
        balance_pcs = pcs.get("BALANCE", [])
        if balance_pcs:
            for b_pc in balance_pcs:
                for c_op in call_ops:
                    for c_pc in pcs.get(c_op, []):
                        if c_pc > b_pc and (c_pc - b_pc) < 10:
                            balance_before_call = 1
                            break

        # Собираем словарь признаков
        features = {
            "total_instructions": total_instructions,
            "unique_instructions": unique_instructions,
            "block_dependent_count": block_dependent_count,
            "block_dependency_index": block_dependent_count / total_instructions,
            **{f"has_{op}": int(op in mnemonic_counter) for op in block_dependent},
            "environmental_instructions_count": environmental_count,
            "environmental_ratio": environmental_count / total_instructions,
            "unique_environmental_ops": unique_env_ops,
            "environmental_complexity": unique_env_ops * (environmental_count / total_instructions),
            "balance_operations": balance_ops,
            "address_operations": mnemonic_counter.get("ADDRESS", 0),
            "caller_operations": caller_ops,
            "origin_operations": origin_ops,
            "callvalue_operations": callvalue_ops,
            "external_dependency_index": (block_dependent_count + balance_ops) / total_instructions,
            "calldata_size_ops": calldata_size,
            "calldata_load_ops": calldata_load,
            "calldata_copy_ops": calldata_copy,
            "total_calldata_ops": calldata_size + calldata_load + calldata_copy,
            "calldata_density": (calldata_size + calldata_load + calldata_copy) / total_instructions,
            "external_call_count": external_call_count,
            "has_external_calls": int(external_call_count > 0),
            "call_value_ops": callvalue_ops,
            "call_gas_limit_ops": gas_ops,
            "potential_reentrancy_pattern": potential_reentrancy,
            "pushes": pushes,
            "pops": pops,
            "stack_imbalance": pushes - pops,
            "stack_operations_ratio": pops / max(1, pushes),
            "stack_underflow_risk": int(pushes - pops < 0),
            "total_gas_cost": total_gas,
            "avg_gas_per_instruction": avg_gas,
            "max_gas_instruction": max_gas,
            "high_gas_instructions": high_gas_count,
            "gas_dos_risk_index": high_gas_count / total_instructions,
            "arithmetic_ops_count": sum(mnemonic_counter.get(op, 0) for op in arithmetic_ops),
            "arithmetic_density": sum(mnemonic_counter.get(op, 0) for op in arithmetic_ops) / total_instructions,
            "unsafe_arithmetic_pattern": unsafe_arith,
            "control_flow_ops": sum(mnemonic_counter.get(op, 0) for op in {"JUMP", "JUMPI", "RETURN", "REVERT", "STOP", "INVALID"}),
            "jumpi_count": mnemonic_counter.get("JUMPI", 0),
            "conditional_branching_ratio": mnemonic_counter.get("JUMPI", 0) / max(1, sum(mnemonic_counter.get(op, 0) for op in {"JUMP", "JUMPI"})),
            "control_flow_complexity": mnemonic_counter.get("JUMPI", 0) ** 2 / total_instructions,
            "caller_based_checks": caller_ops,
            "origin_usage": origin_ops,
            "access_control_ratio": caller_ops / max(1, external_call_count),
            "uses_origin_instead_caller": int(origin_ops > caller_ops),
            "balance_before_external_call": balance_before_call,
            "randomness_ops_count": sum(mnemonic_counter.get(op, 0) for op in randomness_ops),
            "has_bad_randomness_pattern": int(sum(mnemonic_counter.get(op, 0) for op in randomness_ops) > 0),
            "dangerous_ops_count": sum(mnemonic_counter.get(op, 0) for op in dangerous_ops),
            "dangerous_ops_density": sum(mnemonic_counter.get(op, 0) for op in dangerous_ops) / total_instructions,
            "opcode_entropy": entropy(list(mnemonic_counter.values())) if len(mnemonic_counter) > 1 else 0.0,
        }

        # Композитные скоринги
        reentrancy_score = (
            features["external_call_count"] +
            features["call_value_ops"] +
            features["potential_reentrancy_pattern"] +
            features["balance_before_external_call"]
        ) / total_instructions

        frontrunning_score = (
            features["block_dependent_count"] +
            features["external_dependency_index"] +
            features["has_bad_randomness_pattern"]
        ) / total_instructions

        dos_score = (
            features["gas_dos_risk_index"] +
            features["high_gas_instructions"] +
            features["control_flow_complexity"] +
            features["stack_underflow_risk"]
        ) / total_instructions

        arith_score = (
            features["arithmetic_ops_count"] +
            features["unsafe_arithmetic_pattern"] +
            features["stack_underflow_risk"]
        ) / total_instructions

        overall_score = np.mean([
            reentrancy_score, frontrunning_score, dos_score, arith_score,
            features["dangerous_ops_density"], features["external_dependency_index"]
        ])

        features.update({
            "reentrancy_risk_score": reentrancy_score,
            "frontrunning_risk_score": frontrunning_score,
            "dos_risk_score": dos_score,
            "arithmetic_risk_score": arith_score,
            "overall_security_risk_score": overall_score,
            "has_reentrancy_indicators": int(reentrancy_score > 0.1),
            "has_unchecked_external_calls": int(external_call_count > features["jumpi_count"]),
            "has_arithmetic_vulnerabilities": int(features["unsafe_arithmetic_pattern"] > 0),
            "has_access_control_issues": int(features["access_control_ratio"] < 0.2 and external_call_count > 0),
            "has_dos_vulnerabilities": int(features["gas_dos_risk_index"] > 0.1),
        })

        # Гарантируем полный набор признаков
        return {name: features.get(name, 0.0) for name in self.feature_names_}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            bytecodes = X[self.bytecode_column].values
            index = X.index
        else:
            bytecodes = np.asarray(X)
            index = None

        n_jobs = self.n_workers or max(1, mp.cpu_count() - 1)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._extract_features_single)(bc) for bc in bytecodes
        )

        return pd.DataFrame(results, columns=self.feature_names_, index=index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)