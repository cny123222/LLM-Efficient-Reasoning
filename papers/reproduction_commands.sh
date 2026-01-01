#!/bin/bash
# ============================================================================
# Speculative Decoding Paper - Reproduction Script
# ============================================================================
# 
# 使用方法:
#   chmod +x reproduction_commands.sh
#   ./reproduction_commands.sh [experiment_name]
#
# 可选实验:
#   all        - 运行所有实验 (默认)
#   main       - 运行主实验 (全面 benchmark)
#   longseq    - 运行长序列测试
#   specall    - 运行所有 Speculative Decoding 方法综合对比
#   quick      - 快速测试 (100 tokens)
#   full       - 完整测试 (500 tokens, 更多配置)
#
#   treesearch      - Tree V2 参数搜索 (完整, ~30-60分钟)
#   treesearch-quick - Tree V2 参数搜索 (快速)
#   benchmark       - 综合 Benchmark (使用最优 Tree 配置)
#   
#   help       - 显示帮助
#
# ============================================================================

set -e

# 配置路径
PROJECT_ROOT="/mnt/disk1/ljm/LLM-Efficient-Reasoning"
TARGET_MODEL="/mnt/disk1/models/pythia-2.8b"
DRAFT_MODEL="/mnt/disk1/models/pythia-70m"

# 输出目录
RESULTS_DIR="$PROJECT_ROOT/results"
FIGURES_DIR="$PROJECT_ROOT/papers/figures"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_section() { echo -e "${BLUE}========================================${NC}"; }

# 检查环境
check_environment() {
    echo_info "检查环境..."
    [ -d "$PROJECT_ROOT" ] || { echo "项目目录不存在"; exit 1; }
    [ -d "$TARGET_MODEL" ] || { echo "Target 模型不存在"; exit 1; }
    [ -d "$DRAFT_MODEL" ] || { echo "Draft 模型不存在"; exit 1; }
    mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"
    echo_info "环境检查通过!"
}

# 主实验: 全面 benchmark
run_main_experiment() {
    echo_section
    echo_info "运行主实验: 全面性能对比"
    echo_section
    
    cd "$PROJECT_ROOT"
    
    python spec_decode/benchmark_comprehensive.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --max-new-tokens 500 1000 2000 \
        --max-cache-lens 256 512 1024 \
        --k-value 5 \
        --num-samples 3 \
        --output-json "$RESULTS_DIR/benchmark_comprehensive_results.json" \
        --output-plot "$FIGURES_DIR/paper_fig7_comprehensive.png"
    
    echo_info "主实验完成!"
}

# 所有 Spec Decode 方法综合对比 (快速版)
run_spec_all_quick() {
    echo_section
    echo_info "运行所有 Speculative Decoding 方法综合对比 (快速测试)"
    echo_section
    
    cd "$PROJECT_ROOT"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    python papers/benchmark_all_spec_decode.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --max-new-tokens 100 \
        --num-runs 2 \
        --output-json "$RESULTS_DIR/spec_decode_all_quick_${TIMESTAMP}.json" \
        --output-plot "$FIGURES_DIR/spec_decode_all_quick_${TIMESTAMP}.png"
    
    echo_info "快速综合对比完成!"
}

# 所有 Spec Decode 方法综合对比 (完整版)
run_spec_all_full() {
    echo_section
    echo_info "运行所有 Speculative Decoding 方法综合对比 (完整测试)"
    echo_section
    
    cd "$PROJECT_ROOT"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    python papers/benchmark_all_spec_decode.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --max-new-tokens 500 \
        --num-runs 3 \
        --output-json "$RESULTS_DIR/spec_decode_all_full_${TIMESTAMP}.json" \
        --output-plot "$FIGURES_DIR/spec_decode_all_full_${TIMESTAMP}.png"
    
    echo_info "完整综合对比完成!"
}

# 长序列测试
run_longseq_experiment() {
    echo_section
    echo_info "运行长序列测试"
    echo_section
    
    cd "$PROJECT_ROOT"
    
    python spec_decode/benchmark_long_sequence.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --max-new-tokens 500 1000 2000 \
        --max-cache-lens 256 512 1024 \
        --k-value 5 \
        --output-json "$RESULTS_DIR/benchmark_long_seq_results.json" \
        --output-plot "$FIGURES_DIR/paper_fig6_long_seq.png"
    
    echo_info "长序列测试完成!"
}

# Tree V2 参数搜索 (完整版)
run_tree_param_search() {
    echo_section
    echo_info "运行 Tree V2 参数搜索 (完整版)"
    echo_info "预计运行时间: 30-60 分钟"
    echo_section
    
    cd "$PROJECT_ROOT"
    
    python papers/tree_param_search.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --output-dir "$RESULTS_DIR" \
        --num-runs 2
    
    echo_info "Tree V2 参数搜索完成!"
    echo_info "结果保存在 $RESULTS_DIR/tree_param_search_*.json"
}

# Tree V2 参数搜索 (快速版)
run_tree_param_search_quick() {
    echo_section
    echo_info "运行 Tree V2 参数搜索 (快速版)"
    echo_info "预计运行时间: 5-10 分钟"
    echo_section
    
    cd "$PROJECT_ROOT"
    
    python papers/tree_param_search.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --output-dir "$RESULTS_DIR" \
        --num-runs 2 \
        --quick
    
    echo_info "Tree V2 快速参数搜索完成!"
    echo_info "结果保存在 $RESULTS_DIR/tree_param_search_*.json"
}

# 综合 Benchmark (使用最优 Tree 配置)
run_benchmark_all_methods() {
    echo_section
    echo_info "运行综合 Benchmark (所有方法对比)"
    echo_section
    
    cd "$PROJECT_ROOT"
    
    # 查找最新的 tree param search 结果
    TREE_CONFIG=$(ls -t "$RESULTS_DIR"/tree_param_search_*.json 2>/dev/null | head -1)
    
    if [ -n "$TREE_CONFIG" ]; then
        echo_info "使用 Tree 配置: $TREE_CONFIG"
        python papers/benchmark_all_methods.py \
            --target-model "$TARGET_MODEL" \
            --draft-model "$DRAFT_MODEL" \
            --tree-config "$TREE_CONFIG" \
            --max-new-tokens 300 \
            --num-runs 3 \
            --output-dir "$RESULTS_DIR"
    else
        echo_warn "未找到 Tree 参数搜索结果，使用默认配置"
        python papers/benchmark_all_methods.py \
            --target-model "$TARGET_MODEL" \
            --draft-model "$DRAFT_MODEL" \
            --max-new-tokens 300 \
            --num-runs 3 \
            --output-dir "$RESULTS_DIR"
    fi
    
    echo_info "综合 Benchmark 完成!"
}

# 完整 Tree 测试流程 (搜索 + Benchmark)
run_tree_full_pipeline() {
    echo_section
    echo_info "运行完整 Tree 测试流程"
    echo_info "1. Tree V2 参数搜索"
    echo_info "2. 综合 Benchmark 对比"
    echo_section
    
    run_tree_param_search
    run_benchmark_all_methods
    
    echo_section
    echo_info "完整 Tree 测试流程完成!"
    echo_section
}

# 运行所有实验 (不含 spec_all)
run_all() {
    check_environment
    run_main_experiment
    run_longseq_experiment
    
    echo_section
    echo_info "所有实验完成!"
    echo_section
    echo_info "结果文件:"
    echo_info "  - $RESULTS_DIR/benchmark_comprehensive_results.json"
    echo_info "  - $RESULTS_DIR/benchmark_long_seq_results.json"
    echo_info "图表文件:"
    echo_info "  - $FIGURES_DIR/paper_fig6_long_seq.png"
    echo_info "  - $FIGURES_DIR/paper_fig7_comprehensive.png"
}

# 运行完整流程 (包含所有测试)
run_full_suite() {
    check_environment
    run_spec_all_full
    run_main_experiment
    run_longseq_experiment
    
    echo_section
    echo_info "完整测试套件完成!"
    echo_section
    echo_info "所有结果已保存到 $RESULTS_DIR"
    echo_info "所有图表已保存到 $FIGURES_DIR"
}

# 显示帮助
show_help() {
    echo ""
    echo "==============================================="
    echo "  Speculative Decoding Benchmark Suite"
    echo "==============================================="
    echo ""
    echo "用法: $0 [experiment_name]"
    echo ""
    echo "基础实验:"
    echo "  all      - 运行 main + longseq (默认)"
    echo "  main     - 运行主实验 (全面 benchmark)"
    echo "  longseq  - 运行长序列测试"
    echo ""
    echo "快速对比:"
    echo "  specall  - 所有 Spec Decode 方法综合对比 (快速)"
    echo "  quick    - 同上 (快速测试, 100 tokens)"
    echo "  full     - 完整 Spec Decode 对比 (500 tokens)"
    echo ""
    echo "Tree V2 参数搜索:"
    echo "  treesearch       - Tree V2 完整参数搜索 (~30-60分钟)"
    echo "  treesearch-quick - Tree V2 快速参数搜索 (~5-10分钟)"
    echo "  benchmark        - 综合 Benchmark (使用最优 Tree 配置)"
    echo "  tree-pipeline    - 完整流程 (搜索 + Benchmark)"
    echo ""
    echo "完整套件:"
    echo "  suite    - 完整测试套件 (所有实验)"
    echo "  help     - 显示帮助"
    echo ""
    echo "模型配置:"
    echo "  Target: $TARGET_MODEL"
    echo "  Draft:  $DRAFT_MODEL"
    echo ""
    echo "示例:"
    echo "  ./reproduction_commands.sh quick           # 快速综合测试"
    echo "  ./reproduction_commands.sh treesearch-quick # Tree 快速参数搜索"
    echo "  ./reproduction_commands.sh tree-pipeline   # 完整 Tree 测试流程"
    echo "  ./reproduction_commands.sh suite           # 运行所有测试"
    echo ""
}

# 主入口
main() {
    local experiment="${1:-all}"
    
    case "$experiment" in
        all)              run_all ;;
        main)             check_environment; run_main_experiment ;;
        longseq)          check_environment; run_longseq_experiment ;;
        specall|quick)    check_environment; run_spec_all_quick ;;
        full)             check_environment; run_spec_all_full ;;
        treesearch)       check_environment; run_tree_param_search ;;
        treesearch-quick) check_environment; run_tree_param_search_quick ;;
        benchmark)        check_environment; run_benchmark_all_methods ;;
        tree-pipeline)    check_environment; run_tree_full_pipeline ;;
        suite)            run_full_suite ;;
        help|--help|-h)   show_help ;;
        *)                echo "未知实验: $experiment"; show_help; exit 1 ;;
    esac
}

main "$@"
