"""GSM8K flexible extraction: boxed first, then "answer is", then the last number; cut at end-of-turn patterns."""

from swaptest.gsm8k import extract_answer, is_correct, reference_answer


def test_extraction_priority():
    assert extract_answer("so 3 + 4 = 7, then \\boxed{18} and 99 more") == "18"
    assert extract_answer("we get 12 then 13. The answer is 14. Check: 15") == "14"
    assert extract_answer("first 5 then 6 then 7") == "7"
    assert extract_answer("no digits here") is None
    assert extract_answer("$$\n\\boxed{18}\n$$") == "18"


def test_text_is_cut_at_the_first_end_of_turn_pattern():
    assert extract_answer("answer: 4\nQuestion: what is 9?") == "4"
    assert extract_answer("<|start_think|>try 3<|end_think|>The answer is 5<|eot_id|>junk 6") == "5"


def test_numeric_comparison():
    assert is_correct("The answer is $1,200.", "1200")
    assert is_correct("\\boxed{18.00}", "18")
    assert is_correct("so -3", "-3")
    assert not is_correct("The answer is 17", "18")
    assert not is_correct("nothing", "18")
    assert reference_answer("work\n#### 2,400") == "2,400"
