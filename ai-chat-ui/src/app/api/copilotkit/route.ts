import { NextRequest } from "next/server";
import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
  ExperimentalEmptyAdapter,
  LangGraphAgent,
  // langGraphPlatformEndpoint,
} from "@copilotkit/runtime";

const serviceAdapter = new ExperimentalEmptyAdapter();

const runtime = new CopilotRuntime({
  agents: {
    [process.env.NEXT_PUBLIC_COPILOTKIT_AGENT_NAME || "agent"]:
      new LangGraphAgent({
        deploymentUrl: process.env.LANGGRAPH_DEPLOYMENT_URL || "",
        graphId: process.env.NEXT_PUBLIC_COPILOTKIT_AGENT_NAME || "",
        langsmithApiKey: process.env.LANGSMITH_API_KEY || "", // only used in LangGraph Platform deployments
      }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
