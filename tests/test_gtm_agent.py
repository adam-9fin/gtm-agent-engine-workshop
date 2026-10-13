import json
import os
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import gtm_agent as app


class RecordingChatModel(BaseChatModel):
    _bound_tools = PrivateAttr(default_factory=list)
    _prompts = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "recording"

    def bind_tools(self, tools, **kwargs):
        self._bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._prompts.append(messages)
        turn = len(self._prompts)
        if turn == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_current_rep",
                        "args": {},
                        "id": "rep-call",
                        "type": "tool_call",
                    }
                ],
            )
        elif turn == 2:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "send_prospect_email",
                        "args": {
                            "prospect": {"name": "Jamie Prospect", "email": "jamie@example.com"},
                            "subject": "Following up",
                            "body": "Thanks for your time.",
                        },
                        "id": "email-call",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content="Email sent.")
        return ChatResult(generations=[ChatGeneration(message=message)])


class GtmAgentTests(unittest.TestCase):
    def test_agent_uses_app_tools_prompt_and_runtime_injection(self):
        model = RecordingChatModel()
        agent = app._create_gtm_agent(model)
        rep_id = app.REP_IDS[0]["rep_id"]

        result = agent.invoke(
            {"messages": [HumanMessage(content="Email Jamie a follow-up.")]},
            config={"metadata": {"user_id": rep_id}},
        )

        tool_names = {
            tool.name if hasattr(tool, "name") else tool.get("name")
            for tool in model._bound_tools
        }
        self.assertEqual(
            tool_names,
            {
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            },
        )
        system_messages = [message for message in model._prompts[0] if message.type == "system"]
        self.assertEqual([message.content for message in system_messages], [app.SYSTEM_PROMPT])
        self.assertEqual(result["messages"][-1].content, "Email sent.")

        tool_results = {
            message.name: json.loads(message.content)
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        }
        self.assertEqual(tool_results["get_current_rep"]["rep"]["rep_id"], rep_id)
        self.assertEqual(tool_results["send_prospect_email"]["status"], "sent")
        self.assertEqual(
            tool_results["send_prospect_email"]["from"],
            app.data_service.get_rep(rep_id)["email"],
        )

    def test_run_agent_preserves_invocation_and_configuration(self):
        run_id = UUID("b3350e3e-1e56-4561-b97a-0a6ef1ae2ad9")
        agent = Mock()
        agent.invoke.return_value = {"messages": [AIMessage(content="Done.")]}

        with (
            patch.object(app, "gtm_agent", agent),
            patch.object(app.uuid, "uuid4", return_value=run_id),
        ):
            result = app.run_agent(
                "Email prospect LEAD-123",
                user_id="rep-123",
                environment="staging",
                thread_id="thread-123",
            )

        agent.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Email prospect LEAD-123"}]},
            config={
                "run_name": "GTM Assistant",
                "run_id": run_id,
                "metadata": {
                    "thread_id": "thread-123",
                    "user_id": "rep-123",
                    "environment": "staging",
                    "request_intent": "send_email",
                },
            },
        )
        self.assertEqual(
            result,
            {
                "reply": "Done.",
                "messages": [agent.invoke.return_value["messages"][0]],
                "run_id": str(run_id),
            },
        )


if __name__ == "__main__":
    unittest.main()
