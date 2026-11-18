"""Mini NPU Simulator

입력 패턴과 필터를 위치별로 곱하고 모두 더하는 MAC(Multiply-Accumulate) 연산으로
패턴이 Cross(십자가)인지 X인지 판정하는 콘솔 프로그램.
외부 라이브러리 없이 표준 라이브러리만 사용한다.
"""

import json
import os
import sys
import time
import unicodedata

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
# 보너스: 1차원 배열 버전 MAC (메모리 접근 단순화)
# ---------------------------------------------------------------------------
def flatten(matrix):
    """n×n 2차원 배열을 길이 n²의 1차원 배열로 펼친다. (행 우선 순서)"""
    flat = []
    for row in matrix:
        for value in row:
            flat.append(value)
    return flat


def mac_flat(pattern_flat, filter_flat):
    """1차원 배열끼리 MAC 연산. 인덱스 하나(k)로 접근한다.

    2차원 버전은 pattern[i][j]마다 '행 찾기 → 열 찾기' 두 번 인덱싱하지만
    1차원 버전은 pattern[k] 한 번만 인덱싱한다. 곱셈 횟수는 똑같이 n².
    """
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * filter_flat[k]
    return total


# ---------------------------------------------------------------------------
# 보너스: 패턴 생성기
# ---------------------------------------------------------------------------
def generate_cross(n):
    """가운데 행과 가운데 열이 1인 n×n 십자가 패턴을 만든다."""
    m = create_matrix(n)
    mid = n // 2
    for i in range(n):
        set_value(m, mid, i, 1.0)
        set_value(m, i, mid, 1.0)
    return m


def generate_x(n):
    """두 대각선이 1인 n×n X 패턴을 만든다."""
    m = create_matrix(n)
    for i in range(n):
        set_value(m, i, i, 1.0)
        set_value(m, i, n - 1 - i, 1.0)
    return m


GENERATORS = {LABEL_CROSS: generate_cross, LABEL_X: generate_x}


# ---------------------------------------------------------------------------
# 성능 측정
# ---------------------------------------------------------------------------
REPEAT = 10  # 크기별 반복 측정 횟수
PERF_SIZES = [3, 5, 13, 25]


def measure_ms(func, a, b, repeat=REPEAT):
    """func(a, b)를 repeat번 실행한 평균 시간(ms)을 반환한다.

    I/O를 빼고 연산 함수 호출 구간만 잰다. 한 번 실행은 너무 짧아
    타이머 오차가 크므로, repeat번 전체를 한 번에 잰 뒤 횟수로 나눈다.
    측정 전에 한 번 미리 실행(워밍업)해 첫 호출에만 드는 비용을 뺀다.
    """
    func(a, b)  # 워밍업
    start = time.perf_counter()
    for _ in range(repeat):
        func(a, b)
    elapsed = time.perf_counter() - start
    return elapsed / repeat * 1000


def measure_mac_ms(pattern, filt, repeat=REPEAT):
    """2차원 MAC 연산의 평균 시간(ms)."""
    return measure_ms(mac, pattern, filt, repeat)


def text_width(text):
    """콘솔에 표시되는 폭. 한글 등 전각 문자는 2칸으로 센다."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text, width, align="left"):
    """전각 문자 폭을 고려해 text를 width칸에 맞춘다."""
    space = " " * max(0, width - text_width(text))
    return text + space if align == "left" else space + text


def print_table(headers, rows, widths):
    """첫 열은 왼쪽, 나머지 열은 오른쪽 정렬로 표를 출력한다."""
    def line(cells):
        out = []
        for i, cell in enumerate(cells):
            out.append(pad(str(cell), widths[i], "left" if i == 0 else "right"))
        return "".join(out)
    print(line(headers))
    print("-" * sum(widths))
    for row in rows:
        print(line(row))


def print_performance_table(rows):
    """rows: [(n, 평균 ms), ...] 를 크기 / 평균 시간 / 연산 횟수(N²) 표로 출력한다."""
    print_table(
        ["크기", "평균 시간(ms)", "연산 횟수(N²)"],
        [("{}×{}".format(n, n), "{:.4f}".format(ms), n * n) for n, ms in rows],
        [10, 16, 16],
    )


def run_performance_analysis(sizes=PERF_SIZES):
    """크기별로 생성한 Cross 패턴 × Cross 필터의 MAC 시간을 측정해 표로 출력한다.

    모든 크기에서 같은 방식으로 만든 입력을 써서 '크기'만 달라지게 한다.
    반환: [(n, 2D 평균 ms, 1D 평균 ms), ...]
    """
    results = []
    for n in sizes:
        pattern = generate_cross(n)
        filt = generate_cross(n)
        ms_2d = measure_mac_ms(pattern, filt)
        ms_1d = measure_ms(mac_flat, flatten(pattern), flatten(filt))
        results.append((n, ms_2d, ms_1d))
    print_performance_table([(n, ms_2d) for n, ms_2d, _ in results])
    return results


def print_flat_comparison(results):
    """보너스: 2D vs 1D 성능 비교 표 (동일 입력, 동일 반복 횟수)."""
    rows = []
    for n, ms_2d, ms_1d in results:
        ratio = ms_2d / ms_1d if ms_1d > 0 else float("nan")
        rows.append(("{}×{}".format(n, n), "{:.4f}".format(ms_2d), "{:.4f}".format(ms_1d), "{:.2f}x".format(ratio)))
    print_table(["크기", "2D(ms)", "1D(ms)", "2D/1D"], rows, [10, 12, 12, 10])
    print("(2D/1D > 1 이면 1차원 버전이 더 빠름)")


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
    print("{} ({}줄 입력, 공백 구분 / 1행에 cross 또는 x 입력 시 자동 생성)".format(name, n))
    matrix = create_matrix(n)
    r = 0
    while r < n:
        line = input("  {}행> ".format(r + 1))
        if r == 0:
            label = normalize_label(line)
            if label in GENERATORS:
                matrix = GENERATORS[label](n)
                print("  -> {} 패턴 자동 생성".format(label))
                print_matrix(matrix)
                return matrix
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
    perf = run_performance_analysis()

    print_section("[3-1] 보너스: 2D vs 1D 배열 MAC 비교 (평균/{}회)".format(REPEAT))
    print_flat_comparison(perf)

    print_section("[4] 결과 요약")
    print_summary(results)


# ---------------------------------------------------------------------------
# 모드 3 (보너스): 패턴 생성기
# ---------------------------------------------------------------------------
def read_size():
    while True:
        text = input("크기 N 입력 (2 이상 정수, 홀수 권장): ").strip()
        if text.isdigit() and int(text) >= 2:
            return int(text)
        print("입력 형식 오류: 2 이상의 정수를 입력하세요.")


def run_generator_mode():
    n = read_size()
    if n % 2 == 0:
        print("참고: 짝수 크기는 정중앙 칸이 없어 십자가가 한쪽으로 치우칩니다.")
    cross, x = generate_cross(n), generate_x(n)

    print_section("[1] 생성된 패턴 ({}×{})".format(n, n))
    print("Cross:")
    print_matrix(cross)
    print("X:")
    print_matrix(x)

    print_section("[2] 생성 패턴으로 MAC 판정 (필터도 생성기로 만듦)")
    for label, pattern in ((LABEL_CROSS, cross), (LABEL_X, x)):
        s_cross = mac(pattern, cross)
        s_x = mac(pattern, x)
        print("{} 패턴 -> Cross 점수: {} / X 점수: {} / 판정: {}".format(
            label, s_cross, s_x, judge(s_cross, s_x, LABEL_CROSS, LABEL_X)))

    print_section("[3] 성능 분석 (평균/{}회)".format(REPEAT))
    print_performance_table([(n, measure_mac_ms(cross, cross))])


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def select_mode():
    """1 또는 2를 입력받을 때까지 반복한다."""
    while True:
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("3. 패턴 생성기 (보너스)")
        choice = input("선택: ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("1, 2, 3 중 하나를 입력하세요.")


def main():
    print("=== Mini NPU Simulator ===")
    try:
        choice = select_mode()
        if choice == "1":
            run_user_mode()
        elif choice == "3":
            run_generator_mode()
        else:
            # 인자로 다른 JSON 경로를 주면 그 파일을 분석한다 (오류 테스트용)
            path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
            run_json_mode(path)
    except (KeyboardInterrupt, EOFError):
        print("\n입력이 중단되어 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
