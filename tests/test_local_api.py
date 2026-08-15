import unittest

import local_api


class RetrievalAndSafetyTests(unittest.TestCase):
    def first_source(self, question):
        matches = local_api.retrieve(question)
        return matches[0][1]["source"] if matches else None

    def test_common_finance_questions_route_to_the_right_topic(self):
        cases = {
            "What is fixed pay versus variable pay?": "salary-pay-components.txt",
            "What is PF in my payslip?": "provident-fund-pf.txt",
            "Isn't it provisional funds?": "provident-fund-pf.txt",
            "What is CTC versus take-home pay?": "salary-pay-components.txt",
            "What is an emergency fund?": "budgeting-cash-flow.txt",
            "Savings account versus fixed deposit": "bank-accounts-deposits-kyc.txt",
            "Why can a longer EMI loan cost more?": "borrowing-credit-loans.txt",
            "Is a UPI PIN needed to receive money?": "digital-payments-safety.txt",
            "What is the difference between FY and AY?": "income-tax-compliance-workflow.txt",
            "What is AIS and Form 26AS?": "income-tax-compliance-workflow.txt",
            "What is the difference between a tax deduction and a rebate?": "tax-deductions-and-exemptions.txt",
            "Why are mutual funds not guaranteed returns?": "mutual-funds-and-sip-expanded.txt",
            "What should I check in health insurance?": "insurance-policy-and-claims.txt",
            "What is NPS?": "retirement-pension-savings.txt",
            "What are signs of a financial scam?": "financial-scams-and-complaints.txt",
            "What is GST?": "gst-basics.txt",
            "What are mutual funds and SIP for a first-time investor?": "mutual-funds-sip-basics.txt",
            "What are EPF, EPS, EDLI, NPS and PPF?": "retirement-pension-savings.txt",
        }
        for question, expected_source in cases.items():
            with self.subTest(question=question):
                self.assertEqual(self.first_source(question), expected_source)

    def test_abbreviations_and_common_typos_are_normalized(self):
        self.assertIn("provident fund", local_api.normalize_question("what is PF in my payslip"))
        self.assertIn("provident fund", local_api.normalize_question("isn't it provisional funds?"))
        self.assertIn("salary slip", local_api.normalize_question("read my payslip"))
        self.assertIn("26as", local_api.normalize_question("Form 26 AS"))

    def test_unsupported_question_has_no_retrieved_source(self):
        self.assertEqual(local_api.retrieve("What is Bitcoin's price today?"), [])
        self.assertEqual(local_api.retrieve("Explain astrophysics."), [])

    def test_meta_questions_are_deterministic(self):
        answer, sources = local_api.answer_question("How do you work?")
        self.assertIn("retrieve relevant", answer)
        self.assertEqual(sources, [])

    def test_protected_actions_are_refused_before_retrieval(self):
        answer, sources = local_api.answer_question("Can you file my tax return or use my OTP?")
        self.assertEqual(sources, [])
        self.assertIn("cannot file", answer)
        self.assertIn("cannot", answer)

    def test_unknown_questions_do_not_call_the_model(self):
        answer, sources = local_api.answer_question("Explain astrophysics.")
        self.assertIn("reliable source", answer)
        self.assertEqual(sources, [])

    def test_pf_definition_is_deterministic(self):
        answer, sources = local_api.answer_question("What is PF in my payslip?")
        self.assertIn("Provident Fund", answer)
        self.assertNotIn("Personal Funds", answer)
        self.assertEqual(sources[0]["doc_title"], "What Is PF, and How Withdrawal/Transfer Works")

    def test_follow_up_uses_previous_user_question_only(self):
        history = [
            {"role": "user", "content": "What is PF in my payslip?"},
            {"role": "assistant", "content": "PF means Personal Funds."},
        ]
        retrieval_question = local_api.build_retrieval_question("Isn't it provisional funds?", local_api.sanitize_history(history))
        self.assertEqual(local_api.retrieve(retrieval_question)[0][1]["source"], "provident-fund-pf.txt")

    def test_history_drops_assistant_claims_and_is_bounded(self):
        history = [{"role": "assistant", "content": "untrusted answer"}]
        history.extend({"role": "user", "content": f"question {i}"} for i in range(20))
        sanitized = local_api.sanitize_history(history)
        self.assertEqual(len(sanitized), local_api.MAX_HISTORY_MESSAGES)
        self.assertNotIn("untrusted answer", sanitized)


if __name__ == "__main__":
    unittest.main()
