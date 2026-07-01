# Feature Schema

`features/evm_extractor.py` exposes `EVMBytecodeFeatureExtractor`, a
scikit-learn-compatible transformer from raw EVM bytecode to an ordered
tabular feature vector.

## Current Released Schema

The current extractor emits **70 ordered features**.

The paper describes the representation as a lightweight engineered bytecode
feature set. In the released code, this is materialised as:

- 65 count, ratio, complexity, and risk-score features;
- 5 additional binary indicator flags used by the inference schema.

Any trained model must receive features in exactly the order returned by:

```python
extractor.get_feature_names_out()
```

## Feature Groups

| Group | Count |
|---|---:|
| Basic instruction statistics | 2 |
| Block-dependence features | 8 |
| Environmental opcode features | 4 |
| External-dependency features | 6 |
| Calldata features | 5 |
| External-call features | 5 |
| Memory features | 3 |
| Stack features | 5 |
| Gas-analysis features | 5 |
| Arithmetic features | 3 |
| Control-flow features | 4 |
| Access-control features | 4 |
| Advanced-pattern features | 3 |
| Complexity features | 3 |
| Composite risk scores | 5 |
| Binary indicator flags | 5 |
| **Total** | **70** |

## Ordered Feature Names

```text
total_instructions
unique_instructions
block_dependent_count
block_dependency_index
has_TIMESTAMP
has_NUMBER
has_DIFFICULTY
has_GASLIMIT
has_COINBASE
has_BLOCKHASH
environmental_instructions_count
environmental_ratio
unique_environmental_ops
environmental_complexity
balance_operations
address_operations
caller_operations
origin_operations
callvalue_operations
external_dependency_index
calldata_size_ops
calldata_load_ops
calldata_copy_ops
total_calldata_ops
calldata_density
external_call_count
has_external_calls
call_value_ops
call_gas_limit_ops
potential_reentrancy_pattern
reads_from_memory
writes_to_memory
memory_access_ratio
pushes
pops
stack_imbalance
stack_operations_ratio
stack_underflow_risk
total_gas_cost
avg_gas_per_instruction
max_gas_instruction
high_gas_instructions
gas_dos_risk_index
arithmetic_ops_count
arithmetic_density
unsafe_arithmetic_pattern
control_flow_ops
jumpi_count
conditional_branching_ratio
control_flow_complexity
caller_based_checks
origin_usage
access_control_ratio
uses_origin_instead_caller
balance_before_external_call
randomness_ops_count
has_bad_randomness_pattern
dangerous_ops_count
dangerous_ops_density
opcode_entropy
reentrancy_risk_score
frontrunning_risk_score
dos_risk_score
arithmetic_risk_score
overall_security_risk_score
has_reentrancy_indicators
has_unchecked_external_calls
has_arithmetic_vulnerabilities
has_access_control_issues
has_dos_vulnerabilities
```

## Reviewer Note

If future paper revisions continue to use the shorter "65-feature" wording,
the manuscript should define whether the five binary indicator flags are
included in the feature count or treated as derived deployment indicators.
The repository should not leave this implicit.
