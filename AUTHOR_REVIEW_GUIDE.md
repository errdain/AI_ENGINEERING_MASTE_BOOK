# AUTHOR & REVIEW GUIDE

You are contributing to:

AI ENGINEERING MASTER BOOK

Read before working:

1. BOOK_SPECIFICATION.md
2. ROADMAP.md
3. GLOSSARY.md
4. Previous tier summary
5. Assigned tier

DO NOT change the curriculum.

DO NOT renumber concepts.

DO NOT silently introduce new canonical terminology.

If something is missing:

PROPOSE_ADDITION:
<explanation>

Do not automatically insert it.

---

# AUTHOR MODE

Your job:

Teach the assigned concepts.

Required progression:

Intuition
→ Mechanics
→ Mathematics where relevant
→ Implementation
→ Engineering
→ Enterprise
→ Interview

---

# CRITIC MODE

Do NOT rewrite immediately.

Identify:

FACTUAL_ERROR
MISSING_PREREQUISITE
MISSING_CONCEPT
MISLEADING_EXPLANATION
OUTDATED_INFORMATION
CODE_ERROR
MATH_ERROR
ARCHITECTURE_ERROR
SECURITY_GAP
PERFORMANCE_GAP
DUPLICATION
INTERVIEW_WEAKNESS

Severity:

P0 = dangerous/major factual problem
P1 = technically incorrect
P2 = important weakness
P3 = improvement

Return a review report.

---

# CODE REVIEW MODE

For applicable examples:

1. install dependencies
2. execute code
3. run tests
4. validate expected outputs
5. test edge cases
6. test failure conditions
7. verify tensor dimensions
8. check API handling
9. check secret handling
10. report results

Never claim TESTED unless code actually executed.

Use:

TESTED
NOT_TESTED
PARTIALLY_TESTED

---

# FINAL REVIEW MODE

Check:

Curriculum alignment
Technical correctness
Dependency order
Terminology
Code
Architecture
Security
Performance
Cost
Enterprise relevance
Interview quality
Exercises
Revision questions
Memory hook