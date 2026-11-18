"""Mini NPU Simulator

입력 패턴과 필터를 위치별로 곱하고 모두 더하는 MAC(Multiply-Accumulate) 연산으로
패턴이 Cross(십자가)인지 X인지 판정하는 콘솔 프로그램.
외부 라이브러리 없이 표준 라이브러리만 사용한다.
"""

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
EPSILON = 1e-9  # 두 점수 차이가 이 값보다 작으면 동점으로 본다
EPSILON_TEXT = "1e-9"  # 출력용 표기 (1e-09 대신)

LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

# 외부 데이터에서 쓰이는 여러 표기를 내부 표준 라벨로 바꾸는 표
# - expected 값: '+' -> Cross, 'x' -> X
# - filter 키  : 'cross' -> Cross, 'x' -> X
LABEL_ALIASES = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}


# ---------------------------------------------------------------------------
# 데이터 구조: n×n 2차원 배열 (리스트의 리스트)
# ---------------------------------------------------------------------------
def create_matrix(n, fill=0.0):
    """n×n 크기의 2차원 배열을 만들어 fill 값으로 채운다."""
    return [[fill for _ in range(n)] for _ in range(n)]


def get_value(matrix, row, col):
    """특정 위치(row, col)의 값을 읽는다."""
    return matrix[row][col]


def set_value(matrix, row, col, value):
    """특정 위치(row, col)에 값을 저장한다."""
    matrix[row][col] = value


def validate_matrix(matrix, n):
    """matrix가 숫자로 된 n×n 배열인지 검사한다.

    문제가 없으면 None, 문제가 있으면 원인을 설명하는 문자열을 반환한다.
    (예외를 던지지 않으므로 호출한 쪽에서 케이스 단위로 처리할 수 있다.)
    """
    if not isinstance(matrix, list):
        return "배열 형식이 아님"
    if len(matrix) != n:
        return "행 수 불일치 (기대 {}, 실제 {})".format(n, len(matrix))
    for r, row in enumerate(matrix):
        if not isinstance(row, list):
            return "{}행이 배열 형식이 아님".format(r + 1)
        if len(row) != n:
            return "{}행의 열 수 불일치 (기대 {}, 실제 {})".format(r + 1, n, len(row))
        for value in row:
            # bool은 int의 하위 타입이라 따로 걸러낸다
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "{}행에 숫자가 아닌 값 포함: {!r}".format(r + 1, value)
    return None


# ---------------------------------------------------------------------------
# 라벨 정규화
# ---------------------------------------------------------------------------
def normalize_label(raw):
    """외부 표기('+', 'x', 'cross' 등)를 표준 라벨(Cross/X)로 바꾼다.

    알 수 없는 표기면 None을 반환한다.
    """
    if not isinstance(raw, str):
        return None
    return LABEL_ALIASES.get(raw.strip().lower())


# ---------------------------------------------------------------------------
# MAC 연산
# ---------------------------------------------------------------------------
def mac(pattern, filt):
    """패턴과 필터를 같은 위치끼리 곱한 뒤 모두 더한 값(점수)을 반환한다.

    두 배열은 같은 n×n 크기라고 가정한다. (크기 검증은 호출 전에 수행)
    곱셈 횟수 = n × n 이므로 시간 복잡도는 O(N²).
    """
    n = len(pattern)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern[i][j] * filt[i][j]
    return total


# ---------------------------------------------------------------------------
# 점수 비교 / 판정
# ---------------------------------------------------------------------------
def is_tie(score_a, score_b, epsilon=EPSILON):
    """두 점수의 차이가 epsilon보다 작으면 동점으로 본다."""
    return abs(score_a - score_b) < epsilon


def judge(score_a, score_b, label_a, label_b):
    """두 점수를 비교해 더 큰 쪽의 라벨을 반환한다. 동점이면 UNDECIDED."""
    if is_tie(score_a, score_b):
        return LABEL_UNDECIDED
    return label_a if score_a > score_b else label_b


# ---------------------------------------------------------------------------
# 성능 측정
# ---------------------------------------------------------------------------
REPEAT = 10  # 크기별 반복 측정 횟수


def measure_mac_ms(pattern, filt, repeat=REPEAT):
    """MAC 연산을 repeat번 실행한 평균 시간(ms)을 반환한다.

    I/O를 빼고 연산 함수 호출 구간만 잰다. 한 번 실행은 너무 짧아
    타이머 오차가 크므로, repeat번 전체를 한 번에 잰 뒤 횟수로 나눈다.
    """
    start = time.perf_counter()
    for _ in range(repeat):
        mac(pattern, filt)
    elapsed = time.perf_counter() - start
    return elapsed / repeat * 1000


def print_performance_table(rows):
    """rows: [(n, 평균 ms), ...] 를 크기 / 평균 시간 / 연산 횟수(N²) 표로 출력한다."""
    print("{:<10}{:>16}{:>12}".format("크기", "평균 시간(ms)", "연산 횟수"))
    print("-" * 40)
    for n, avg_ms in rows:
        size = "{}×{}".format(n, n)
        print("{:<10}{:>16.4f}{:>12}".format(size, avg_ms, n * n))


# ---------------------------------------------------------------------------
# 콘솔 출력 도우미
# ---------------------------------------------------------------------------
def print_section(title):
    print()
    print("#" + "-" * 40)
    print("# " + title)
    print("#" + "-" * 40)


def print_matrix(matrix):
    for row in matrix:
        print("  " + " ".join(str(v) for v in row))


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3×3)
# ---------------------------------------------------------------------------
def parse_row(line, n):
    """한 줄을 공백 기준으로 나눠 숫자 n개의 리스트로 바꾼다.

    성공하면 (리스트, None), 실패하면 (None, 오류 메시지)를 반환한다.
    """
    tokens = line.split()
    if len(tokens) != n:
        return None, "입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요. (입력된 개수: {})".format(n, len(tokens))
    row = []
    for token in tokens:
        try:
            row.append(float(token))
        except ValueError:
            return None, "입력 형식 오류: '{}'은(는) 숫자가 아닙니다. 숫자만 입력하세요.".format(token)
    return row, None


def read_matrix(name, n=3):
    """n줄을 한 줄씩 입력받아 n×n 배열을 만든다. 잘못된 줄은 그 줄만 다시 입력받는다."""
    print("{} ({}줄 입력, 공백 구분)".format(name, n))
    matrix = create_matrix(n)
    r = 0
    while r < n:
        line = input("  {}행> ".format(r + 1))
        row, error = parse_row(line, n)
        if error:
            print("  " + error)
            continue  # 같은 행을 다시 입력받는다
        for c in range(n):
            set_value(matrix, r, c, row[c])
        r += 1
    return matrix


def run_user_mode():
    n = 3

    print_section("[1] 필터 입력")
    filter_a = read_matrix("필터 A", n)
    print()
    filter_b = read_matrix("필터 B", n)

    print("\n저장된 필터 A:")
    print_matrix(filter_a)
    print("저장된 필터 B:")
    print_matrix(filter_b)

    print_section("[2] 패턴 입력")
    pattern = read_matrix("패턴", n)

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    avg_ms = measure_mac_ms(pattern, filter_a)
    result = judge(score_a, score_b, "A", "B")

    print_section("[3] MAC 결과")
    print("A 점수:", repr(score_a))
    print("B 점수:", repr(score_b))
    print("연산 시간(평균/{}회): {:.4f} ms".format(REPEAT, avg_ms))
    if result == LABEL_UNDECIDED:
        print("판정: 판정 불가 (|A-B| < {})".format(EPSILON_TEXT))
    else:
        print("판정:", result)

    print_section("[4] 성능 분석 (평균/{}회)".format(REPEAT))
    print_performance_table([(n, avg_ms)])


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")


def parse_size_key(key, with_index=False):
    """'size_{N}' 또는 'size_{N}_{idx}' 형태의 키에서 N을 꺼낸다.

    형식이 맞지 않으면 None을 반환한다.
    """
    if not isinstance(key, str):
        return None
    parts = key.split("_")
    expected_len = 3 if with_index else 2
    if len(parts) != expected_len or parts[0] != "size":
        return None
    if not all(p.isdigit() for p in parts[1:]):
        return None
    n = int(parts[1])
    return n if n > 0 else None


def load_json(path):
    """JSON 파일을 읽는다. 성공하면 (데이터, None), 실패하면 (None, 오류 메시지)."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "파일을 찾을 수 없음: {}".format(path)
    except json.JSONDecodeError as e:
        return None, "JSON 형식 오류: {}".format(e)


def load_filters(data):
    """data['filters']를 읽어 {N: {'Cross': 배열, 'X': 배열}} 형태로 반환한다.

    필터 키('cross', 'x')는 표준 라벨로 정규화한다.
    문제가 있는 크기는 건너뛰고 이유를 출력한다. (해당 크기 패턴은 나중에 FAIL 처리)
    """
    filters = {}
    raw_filters = data.get("filters") if isinstance(data, dict) else None
    if not isinstance(raw_filters, dict):
        print("✗ filters 항목이 없거나 형식이 잘못됨")
        return filters

    for size_key, group in raw_filters.items():
        n = parse_size_key(size_key)
        if n is None:
            print("✗ {:<8} 키 형식 오류 (size_N 형태가 아님)".format(size_key))
            continue
        if not isinstance(group, dict):
            print("✗ {:<8} 필터 묶음 형식 오류".format(size_key))
            continue

        loaded = {}
        error = None
        for raw_label, matrix in group.items():
            label = normalize_label(raw_label)
            if label is None:
                error = "알 수 없는 필터 라벨: {!r}".format(raw_label)
                break
            problem = validate_matrix(matrix, n)
            if problem:
                error = "{} 필터 {}".format(label, problem)
                break
            loaded[label] = matrix

        if error is None:
            missing = [lb for lb in (LABEL_CROSS, LABEL_X) if lb not in loaded]
            if missing:
                error = "필터 누락: {}".format(", ".join(missing))

        if error:
            print("✗ {:<8} 필터 로드 실패 ({})".format(size_key, error))
        else:
            filters[n] = loaded
            print("✓ {:<8} 필터 로드 완료 (Cross, X)".format(size_key))
    return filters


def analyze_pattern(case_id, case, filters):
    """패턴 1개를 검증·판정하고 결과 딕셔너리를 반환한다. 예외를 밖으로 던지지 않는다.

    반환: {'id', 'passed', 'reason', 'cross', 'x', 'result', 'expected'}
    """
    result = {"id": case_id, "passed": False, "reason": "",
              "cross": None, "x": None, "result": None, "expected": None}

    n = parse_size_key(case_id, with_index=True)
    if n is None:
        result["reason"] = "키 형식 오류 (size_N_idx 형태가 아님)"
        return result
    if not isinstance(case, dict):
        result["reason"] = "패턴 항목 형식 오류"
        return result

    expected = normalize_label(case.get("expected"))
    result["expected"] = expected
    if expected is None:
        result["reason"] = "알 수 없는 expected 값: {!r}".format(case.get("expected"))
        return result

    if n not in filters:
        result["reason"] = "size_{} 필터가 없음".format(n)
        return result

    problem = validate_matrix(case.get("input"), n)
    if problem:
        result["reason"] = "패턴 크기/형식 불일치 - {}".format(problem)
        return result

    pattern = case["input"]
    score_cross = mac(pattern, filters[n][LABEL_CROSS])
    score_x = mac(pattern, filters[n][LABEL_X])
    decision = judge(score_cross, score_x, LABEL_CROSS, LABEL_X)
    result.update(cross=score_cross, x=score_x, result=decision)

    if decision == LABEL_UNDECIDED:
        result["reason"] = "동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| < {})".format(EPSILON_TEXT)
    elif decision != expected:
        result["reason"] = "판정 {}이(가) expected {}와(과) 다름".format(decision, expected)
    else:
        result["passed"] = True
    return result


def print_pattern_result(r):
    print("--- {} ---".format(r["id"]))
    if r["result"] is None:  # 연산 전에 검증 단계에서 실패
        print("FAIL ({})".format(r["reason"]))
        return
    print("Cross 점수:", repr(r["cross"]))
    print("X 점수:", repr(r["x"]))
    status = "PASS" if r["passed"] else "FAIL"
    if r["result"] == LABEL_UNDECIDED:
        status += " (동점 규칙)"
    print("판정: {} | expected: {} | {}".format(r["result"], r["expected"], status))


def print_summary(results):
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    print("총 테스트: {}개".format(len(results)))
    print("통과: {}개".format(len(passed)))
    print("실패: {}개".format(len(failed)))
    if failed:
        print("\n실패 케이스:")
        for r in failed:
            print("- {}: {}".format(r["id"], r["reason"]))


def run_json_mode(path=DEFAULT_DATA_PATH):
    data, error = load_json(path)
    if error:
        print(error)
        return

    print_section("[1] 필터 로드")
    filters = load_filters(data)

    print_section("[2] 패턴 분석 (라벨 정규화 적용)")
    patterns = data.get("patterns") if isinstance(data, dict) else None
    results = []
    if not isinstance(patterns, dict):
        print("patterns 항목이 없거나 형식이 잘못됨")
    else:
        for case_id, case in patterns.items():
            r = analyze_pattern(case_id, case, filters)
            print_pattern_result(r)
            results.append(r)

    print_section("[3] 성능 분석 (평균/{}회)".format(REPEAT))
    print("(4단계에서 구현 예정)")

    print_section("[4] 결과 요약")
    print_summary(results)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def select_mode():
    """1 또는 2를 입력받을 때까지 반복한다."""
    while True:
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        choice = input("선택: ").strip()
        if choice in ("1", "2"):
            return choice
        print("1 또는 2를 입력하세요.")


def main():
    print("=== Mini NPU Simulator ===")
    try:
        choice = select_mode()
        if choice == "1":
            run_user_mode()
        else:
            # 인자로 다른 JSON 경로를 주면 그 파일을 분석한다 (오류 테스트용)
            path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
            run_json_mode(path)
    except (KeyboardInterrupt, EOFError):
        print("\n입력이 중단되어 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
