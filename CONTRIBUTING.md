# Contributing to SmartKNN

Thank you for your interest in **SmartKNN**.

SmartKNN is a **performance-first, CPU-efficient nearest-neighbor engine** designed for deterministic behavior, interpretability, and real-world benchmarking. Contributions are welcome, but this project prioritizes **engineering discipline and measurable impact** over feature quantity.

Please read this document carefully before opening an issue or submitting a pull request.

---

## Project Philosophy

SmartKNN is guided by the following principles:

- **Predictable performance over heuristics**
- **CPU-first design with minimal hidden costs**
- **Deterministic and explainable behavior**
- **Benchmarks and evidence over opinions**

Any contribution should align with these principles. Proposals that conflict with them are unlikely to be accepted.

---

## Ways to Contribute

You can contribute to SmartKNN in several meaningful ways:

- Reporting **reproducible bugs** with sufficient context
- Improving **documentation, clarity, or explanations**
- Optimizing **performance, memory usage, or latency**
- Adding **benchmarks or evaluation coverage**
- Proposing **well-justified architectural improvements**

Low-effort changes, cosmetic refactors, or speculative features without data may be declined.

---

## Before Opening an Issue

Before opening an issue, please ensure that:

- The issue is **specific and reproducible**
- You have reviewed existing issues to avoid duplication
- Benchmark-related issues include dataset characteristics and hardware context
- Performance regressions include measurable comparisons

Issues without sufficient detail may be closed for clarification.

---

## Pull Request Guidelines

All pull requests should meet the following expectations:

- The purpose of the change is **clearly explained**
- The change aligns with SmartKNN’s design philosophy
- Performance or memory impact is considered and stated
- Benchmarks or reasoning are provided where applicable
- Unrelated changes are avoided

Large or architectural changes should be discussed in an issue before submitting a pull request.

---

## Performance and Benchmarks

SmartKNN is a performance-sensitive project.

Contributions that affect:

- Query latency
- Memory layout
- Index construction
- Candidate selection
- Distance computation

must clearly explain **why the change is beneficial** and under what conditions. Benchmark-driven contributions are strongly preferred.

---

## Scope and Non-Goals

To keep the project focused, the following are generally considered out of scope:

- GPU-only implementations
- Black-box heuristics without explanation
- Excessive abstraction that reduces transparency
- Features that significantly increase complexity without measurable benefit

This list is not exhaustive, but it reflects the intended direction of the project.

---

## Communication and Conduct

Please keep discussions **technical, respectful, and focused**.

SmartKNN values:

- Clear technical reasoning
- Constructive feedback
- Respect for differing approaches backed by data

Unproductive or hostile behavior may result in issues or pull requests being closed.

---

## Final Notes

Not all contributions can be accepted, even if well-intentioned. Rejections are based on **project direction and technical fit**, not effort.

Thank you for helping improve SmartKNN.
