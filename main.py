"""Mini NPU Simulator

입력 패턴과 필터를 위치별로 곱하고 모두 더하는 MAC(Multiply-Accumulate) 연산으로
패턴이 Cross(십자가)인지 X인지 판정하는 콘솔 프로그램.
외부 라이브러리 없이 표준 라이브러리만 사용한다.
"""

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
EPSILON = 1e-9  # 두 점수 차이가 이 값보다 작으면 동점으로 본다

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
# 실행 (1단계: 핵심 함수 동작 확인용 데모 — 이후 단계에서 메뉴로 교체 예정)
# ---------------------------------------------------------------------------
def main():
    cross = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    x = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]

    print("[1] MAC 연산 (미션 소개의 Case 1, 2)")
    print("  십자가 입력 x 십자가 필터 =", mac(cross, cross))
    print("  십자가 입력 x X 필터     =", mac(cross, x))
    print("  판정:", judge(mac(cross, cross), mac(cross, x), LABEL_CROSS, LABEL_X))

    print("\n[2] 부동소수점 오차와 epsilon 비교")
    total = 0.0
    for _ in range(9):
        total += 0.1
    print("  0.1을 9번 더한 값 =", repr(total))
    print("  0.9               =", repr(0.9))
    print("  == 비교           :", total == 0.9)
    print("  epsilon 비교(동점):", is_tie(total, 0.9))
    print("  판정              :", judge(total, 0.9, LABEL_CROSS, LABEL_X))

    print("\n[3] 라벨 정규화")
    for raw in ["+", "x", "cross", " X ", "Cross", "?"]:
        print("  {!r:>8} -> {}".format(raw, normalize_label(raw)))

    print("\n[4] 데이터 구조 / 검증")
    m = create_matrix(3)
    set_value(m, 1, 1, 1.0)
    print("  3x3 생성 후 (1,1)에 1.0 저장 ->", get_value(m, 1, 1), m)
    print("  정상 3x3       :", validate_matrix(cross, 3))
    print("  5x5로 검증     :", validate_matrix(cross, 5))
    print("  열 수 틀림     :", validate_matrix([[0, 1], [1, 1, 1], [0, 1, 0]], 3))
    print("  숫자 아님      :", validate_matrix([[0, "a", 0], [1, 1, 1], [0, 1, 0]], 3))


if __name__ == "__main__":
    main()
