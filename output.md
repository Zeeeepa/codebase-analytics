# 📊 Repository Analysis Report 📊
==================================================

## 📁 Repository Overview
**Repository:** Zeeeepa/codebase-analytics
**Description:** Analytics for codebase maintainability and complexity
**Analysis Date:** 2025-07-11

### 📊 Summary Statistics
- **📁 Files:** 4
- **🔄 Functions:** 31
- **🎯 Entry Points:** 8
- **🚨 Total Issues:** 12
- **⚠️ Critical Issues:** 3
- **👉 Major Issues:** 9
- **🔍 Minor Issues:** 0

---

## 🌳 Repository Structure

```
Zeeeepa/codebase-analytics/
└── 📁 backend/ [⚠️ Critical: 3] [👉 Major: 9]
    ├── 📄 __init__.py 
    ├── 📄 analysis.py [⚠️ Critical: 1] [👉 Major: 8]
    ├── 📄 api.py [⚠️ Critical: 2] [👉 Major: 1]
    └── 📄 models.py 
```

---

## 🌟 Most Important Functions & Entry Points

1. **🎯 get_function_context_enhanced** (Score: 80)
   - **File:** backend/analysis.py
   - **Entry Point:** True
   - **Usage Count:** 3
   - **Issues:** 0
   - **Halstead Volume:** 765.1

2. **🎯 get_max_call_chain_enhanced** (Score: 67)
   - **File:** backend/analysis.py
   - **Entry Point:** True
   - **Usage Count:** 2
   - **Issues:** 0
   - **Halstead Volume:** 701.8

3. **🎯 get_comprehensive_analysis_report** (Score: 67)
   - **File:** backend/analysis.py
   - **Entry Point:** True
   - **Usage Count:** 1
   - **Issues:** 1
   - **Halstead Volume:** 1563.2

4. **🎯 get_repository_structure_with_analysis** (Score: 67)
   - **File:** backend/analysis.py
   - **Entry Point:** True
   - **Usage Count:** 1
   - **Issues:** 0
   - **Halstead Volume:** 890.3

5. **🎯 get_function_issues_with_context** (Score: 63)
   - **File:** backend/analysis.py
   - **Entry Point:** True
   - **Usage Count:** 1
   - **Issues:** 1
   - **Halstead Volume:** 2294.2

6. **🎯 get_repo_description** (Score: 54)
   - **File:** backend/api.py
   - **Entry Point:** True
   - **Usage Count:** 1
   - **Issues:** 0
   - **Halstead Volume:** 664.9

7. **🎯 get_codebase_summary** (Score: 51)
   - **File:** backend/analysis.py
   - **Entry Point:** True
   - **Usage Count:** 0
   - **Issues:** 0
   - **Halstead Volume:** 484.3

8. **🔧 format_issue_counts** (Score: 47)
   - **File:** backend/analysis.py
   - **Entry Point:** False
   - **Usage Count:** 5
   - **Issues:** 0
   - **Halstead Volume:** 498.6

9. **🎯 fastapi_app_modal** (Score: 42)
   - **File:** backend/api.py
   - **Entry Point:** True
   - **Usage Count:** 0
   - **Issues:** 0
   - **Halstead Volume:** 62.9

10. **🔧 analyze_file_for_tree** (Score: 41)
   - **File:** backend/analysis.py
   - **Entry Point:** False
   - **Usage Count:** 2
   - **Issues:** 0
   - **Halstead Volume:** 1262.6

## 🎯 Critical Entry Points

### 🚀 **get_function_context_enhanced**
- **File:** backend/analysis.py
- **Importance Score:** 80/100
- **Usage Count:** 3
- **Call Count:** 20
- **Dependencies:** 7
- **Issues:** 0

### 🚀 **get_max_call_chain_enhanced**
- **File:** backend/analysis.py
- **Importance Score:** 67/100
- **Usage Count:** 2
- **Call Count:** 15
- **Dependencies:** 1
- **Issues:** 0

### 🚀 **get_comprehensive_analysis_report**
- **File:** backend/analysis.py
- **Importance Score:** 67/100
- **Usage Count:** 1
- **Call Count:** 16
- **Dependencies:** 2
- **Issues:** 1

### 🚀 **get_repository_structure_with_analysis**
- **File:** backend/analysis.py
- **Importance Score:** 67/100
- **Usage Count:** 1
- **Call Count:** 10
- **Dependencies:** 2
- **Issues:** 0

### 🚀 **get_function_issues_with_context**
- **File:** backend/analysis.py
- **Importance Score:** 63/100
- **Usage Count:** 1
- **Call Count:** 15
- **Dependencies:** 2
- **Issues:** 1

### 🚀 **get_repo_description**
- **File:** backend/api.py
- **Importance Score:** 54/100
- **Usage Count:** 1
- **Call Count:** 7
- **Dependencies:** 1
- **Issues:** 0

### 🚀 **get_codebase_summary**
- **File:** backend/analysis.py
- **Importance Score:** 51/100
- **Usage Count:** 0
- **Call Count:** 7
- **Dependencies:** 3
- **Issues:** 0

### 🚀 **fastapi_app_modal**
- **File:** backend/api.py
- **Importance Score:** 42/100
- **Usage Count:** 0
- **Call Count:** 2
- **Dependencies:** 4
- **Issues:** 0

## 🚨 Critical Issues & Error Analysis

### ⚠️ **create_health_dashboard**
- **File:** backend/analysis.py
- **Lines:** 0-0
- **Importance Score:** 26
- **Critical Issues:**
  - Potential null reference in 'create_health_dashboard'
    - **Fix:** Add null check before using .get() result

### ⚠️ **_calculate_coupling_cohesion_metrics**
- **File:** backend/analysis.py
- **Lines:** 0-0
- **Importance Score:** 28
- **Major Issues:**
  - Function '_calculate_coupling_cohesion_metrics' may be missing return statement
    - **Fix:** Add explicit return statement

### ⚠️ **_detect_function_importance**
- **File:** backend/analysis.py
- **Lines:** 0-0
- **Importance Score:** 24
- **Major Issues:**
  - Function '_detect_function_importance' may be missing return statement
    - **Fix:** Add explicit return statement

### ⚠️ **_build_call_chains**
- **File:** backend/analysis.py
- **Lines:** 0-0
- **Importance Score:** 22
- **Major Issues:**
  - Function '_build_call_chains' may be missing return statement
    - **Fix:** Add explicit return statement

### ⚠️ **get_function_issues_with_context**
- **File:** backend/analysis.py
- **Lines:** 0-0
- **Importance Score:** 63
- **Major Issues:**
  - Function 'get_function_issues_with_context' is too long (72 lines)
    - **Fix:** Break down into smaller functions

---
**Analysis Engine:** Graph-sitter with comprehensive AST analysis
**Report Generated:** 2025-07-11 11:55:00 UTC