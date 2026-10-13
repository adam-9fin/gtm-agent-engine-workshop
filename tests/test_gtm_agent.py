import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import langchain_openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class _ScriptedChatModel(BaseChatModel):
    model: str = "test-model"
    temperature: float = 0
    call_count: int = 0
    saw_blocked_result: bool = False

    @property
    def _llm_type(self):
        return "scripted-test-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.call_count == 0:
            response = AIMessage(
                content="",
                tool_calls=[{"name": "get_current_rep", "args": {}, "id": "rep", "type": "tool_call"}],
            )
        elif self.call_count == 1:
            response = AIMessage(
                content="",
                tool_calls=[{
                    "name": "send_prospect_email",
                    "args": {
                        "prospect": {
                            "prospect_id": "PROSPECT-1",
                            "name": "Alex Prospect",
                            "email": "alex@example.com",
                        },
                        "subject": "Following up",
                        "body": "Hello Alex",
                    },
                    "id": "send",
                    "type": "tool_call",
                }],
            )
        else:
            self.saw_blocked_result = any(
                isinstance(message, ToolMessage) and "blocked" in str(message.content)
                for message in messages
            )
            response = AIMessage(
                content=(
                    "I did not send the email because the prospect is disqualified."
                    if self.saw_blocked_result
                    else "The email was sent."
                )
            )
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema):
        return self


with patch.object(langchain_openai, "ChatOpenAI", _ScriptedChatModel):
    import gtm_agent.gtm_agent as agent_module

    from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {
            "prospect_id": "PROSPECT-1",
            "name": "Alex Prospect",
            "email": "alex@example.com",
        }
        self.rep = {"name": "Sam Rep", "email": "sam@example.com"}
        self.runtime = SimpleNamespace(config={"metadata": {}})

    def send_email(self):
        return send_prospect_email.func(
            self.prospect,
            "Following up",
            "Hello Alex",
            self.runtime,
            from_rep=self.rep,
        )

    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_disqualified_prospect_is_blocked_without_sending(self, get_record):
        get_record.return_value = {"disqualified": True}

        result = self.send_email()

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is disqualified; email not sent."},
        )

    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_qualified_prospect_is_sent(self, get_record):
        get_record.return_value = {"disqualified": False}

        result = self.send_email()

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], self.prospect["email"])

    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record")
    def test_missing_source_record_keeps_existing_send_behavior(self, get_record):
        get_record.return_value = None

        result = self.send_email()

        self.assertEqual(result["status"], "sent")

    def test_prompt_requires_honest_blocked_send_report(self):
        self.assertIn("do not send emails to disqualified prospects", SYSTEM_PROMPT)
        self.assertIn("If a send is blocked, tell the rep it was not sent", SYSTEM_PROMPT)

    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value={"disqualified": True})
    def test_agent_reports_blocked_send_without_claiming_success(self, get_record):
        model = agent_module.agent_model
        model.call_count = 0
        model.saw_blocked_result = False

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}):
            result = agent_module.gtm_agent.invoke(
                {"messages": [{"role": "user", "content": "Email Alex at alex@example.com."}]},
                config={"metadata": {"user_id": "rep-001"}},
            )

        reply = result["messages"][-1].content
        self.assertTrue(model.saw_blocked_result)
        self.assertIn("did not send", reply)
        self.assertNotIn("email was sent", reply.lower())
        get_record.assert_called_once_with("PROSPECT-1")


if __name__ == "__main__":
    unittest.main()
