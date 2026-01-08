from typing import TypedDict, Literal, Any

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from app.core.logger import logger
from app.services.retrieval import retrieval_service
from app.services.llm import llm_service


class GraphState(TypedDict):
    query: str
    query_type: Literal["search", "question", "greeting"]
    documents: list[Document]
    context: str
    answer: str
    sources: list[dict[str, Any]]
    error: str | None
    top_k: int
    temperature: float


class QueryRouter:
    GREETING_KEYWORDS = ["hello", "hey"]
    SEARCH_KEYWORDS = ["find", "search", "show", "list"]

    @staticmethod
    def route(state: GraphState) -> GraphState:
        query = state["query"].lower()
        if any(keyword in query for keyword in QueryRouter.GREETING_KEYWORDS):
            logger.info("Query type: greeting")
            state["query_type"] = "greeting"
            return state

        if any(keyword in query for keyword in QueryRouter.SEARCH_KEYWORDS):
            logger.info("Query type: search")
            state["query_type"] = "search"
            return state

        logger.info("Query type: question")
        state["query_type"] = "question"
        return state


class GraphNodes:
    @staticmethod
    async def search_node(state: GraphState) -> GraphState:
        try:
            logger.info(f"Searching for: '{state['query']}'")

            documents = await retrieval_service.search(
                query=state["query"], k=state.get("top_k", 4), score_threshold=0.0
            )

            state["documents"] = documents
            logger.info(f"Found {len(documents)} documents")

            return state

        except Exception as e:
            logger.error(f"Search node failed: {e}")
            state["error"] = str(e)
            state["documents"] = []
            return state

    @staticmethod
    def format_context_node(state: GraphState) -> GraphState:
        try:
            documents = state.get("documents", [])

            if not documents:
                state["context"] = "No relevant information found."
                state["sources"] = []
                return state

            context = retrieval_service.format_context(documents)
            sources = retrieval_service.get_sources(documents)

            state["context"] = context
            state["sources"] = sources

            logger.info(f"Context formatted: {len(context)} chars")

            return state

        except Exception as e:
            logger.error(f"Format context node failed: {e}")
            state["error"] = str(e)
            state["context"] = ""
            state["sources"] = []
            return state

    @staticmethod
    async def generate_answer_node(state: GraphState) -> GraphState:
        try:
            logger.info("Generating answer...")

            response = await llm_service.generate_answer(
                question=state["query"],
                context=state.get("context", ""),
                temperature=state.get("temperature", 0.7),
            )

            state["answer"] = response["answer"]
            logger.info("Answer generated")

            return state

        except Exception as e:
            logger.error(f"Generate answer node failed: {e}")
            state["error"] = str(e)
            state["answer"] = f"Error generating answer: {str(e)}"
            return state

    @staticmethod
    async def greeting_node(state: GraphState) -> GraphState:
        state["answer"] = (
            "Hello! I'm a RAG-powered assistant. "
            "I can help you find information in the uploaded documents. "
            "Just ask me a question!"
        )
        state["sources"] = []
        state["context"] = ""

        logger.info("Greeting response generated")

        return state

    @staticmethod
    def search_only_node(state: GraphState) -> GraphState:
        documents = state.get("documents", [])

        if not documents:
            state["answer"] = "No documents found matching your query."
        else:
            results = []
            for i, doc in enumerate(documents, 1):
                filename = doc.metadata.get("filename", "Unknown")
                score = doc.metadata.get("relevance_score", 0)
                results.append(
                    f"{i}. {filename} (relevance: {score:.2f})\n"
                    f"   {doc.page_content[:200]}..."
                )

            state["answer"] = "Found documents:\n\n" + "\n\n".join(results)

        logger.info("Search results formatted")

        return state


class LangGraphService:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        workflow.add_node("route", QueryRouter.route)
        workflow.add_node("search", GraphNodes.search_node)
        workflow.add_node("format_context", GraphNodes.format_context_node)
        workflow.add_node("generate_answer", GraphNodes.generate_answer_node)
        workflow.add_node("greeting", GraphNodes.greeting_node)
        workflow.add_node("search_only", GraphNodes.search_only_node)

        workflow.set_entry_point("route")

        def route_query(state: GraphState) -> str:
            query_type = state.get("query_type", "question")

            if query_type == "greeting":
                return "greeting"
            elif query_type == "search":
                return "search_for_list"
            else:
                return "search_for_qa"

        workflow.add_conditional_edges(
            "route",
            route_query,
            {
                "greeting": "greeting",
                "search_for_list": "search",
                "search_for_qa": "search",
            },
        )

        workflow.add_edge("greeting", END)

        workflow.add_conditional_edges(
            "search",
            lambda s: "format" if s.get("query_type") == "question" else "list",
            {"format": "format_context", "list": "search_only"},
        )

        workflow.add_edge("search_only", END)
        workflow.add_edge("format_context", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    async def process(
        self, query: str, top_k: int = 4, temperature: float = 0.7
    ) -> dict[str, Any]:
        try:
            logger.info(f"Processing query through graph: '{query[:50]}...'")
            initial_state: GraphState = {
                "query": query,
                "query_type": "question",
                "documents": [],
                "context": "",
                "answer": "",
                "sources": [],
                "error": None,
                "top_k": top_k,
                "temperature": temperature,
            }

            result = await self.graph.ainvoke(initial_state)

            logger.info("Graph execution completed")

            return {
                "answer": result.get("answer", ""),
                "question": query,
                "query_type": result.get("query_type", "question"),
                "sources": result.get("sources", []),
                "context_used": bool(result.get("context")),
                "error": result.get("error"),
            }

        except Exception as e:
            logger.error(f"Graph processing failed: {e}", exc_info=True)
            return {
                "answer": f"Error processing query: {str(e)}",
                "question": query,
                "sources": [],
                "error": str(e),
            }


langgraph_service = LangGraphService()
