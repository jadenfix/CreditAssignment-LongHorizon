from cts.rl.verifier import CodeVerifier, MathVerifier, extract_final_number


def test_extract_final_number():
    assert extract_final_number("the answer is 42") == "42"
    assert extract_final_number("3 + 5 = 8 ; final 12.5") == "12.5"
    assert extract_final_number("no numbers here") is None


def test_math_verifier():
    v = MathVerifier(gold={"t": "5"})
    assert v.score("t", "answer is 5") == 1.0
    assert v.score("t", "answer is 6") == 0.0
    assert v.score("missing", "5") == 0.0


def test_code_verifier_pass_and_fail():
    v = CodeVerifier(
        tests={
            "ok": ["assert add(1,2) == 3"],
            "bad": ["assert add(1,2) == 3"],
        }
    )
    assert v.score("ok", "def add(a,b):\n    return a+b") == 1.0
    assert v.score("bad", "def add(a,b):\n    return a-b") == 0.0
