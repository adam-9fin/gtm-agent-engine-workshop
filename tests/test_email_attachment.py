import importlib
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

gtm_agent_module = importlib.import_module("gtm_agent.gtm_agent")
SYSTEM_PROMPT = gtm_agent_module.SYSTEM_PROMPT
send_prospect_email = gtm_agent_module.send_prospect_email


class EmailAttachmentTests(unittest.TestCase):
    def test_pricing_deck_request_requires_attachment_disclosure(self):
        self.assertIn("Files and collateral cannot be attached", SYSTEM_PROMPT)
        self.assertIn("tell the rep it could not be attached", SYSTEM_PROMPT)
        self.assertIn("do not say or imply it is attached, included, or sent", SYSTEM_PROMPT)
        self.assertIn("clearly marked placeholder link", SYSTEM_PROMPT)
        self.assertIn("cannot attach files", send_prospect_email.description)
        self.assertIn("never say anything is attached", send_prospect_email.description)

    def test_ordinary_email_can_still_be_sent(self):
        result = send_prospect_email.func(
            prospect={"name": "Sofia Rossi", "email": "sofia@example.com"},
            subject="Scheduling a call",
            body="Hi Sofia, are you available next week?",
            runtime=SimpleNamespace(config={"metadata": {}}),
            from_rep={"name": "Sara Brown", "email": "sara@example.com"},
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], "Hi Sofia, are you available next week?")
        self.assertNotIn("attach", result["body"].lower())


if __name__ == "__main__":
    unittest.main()
