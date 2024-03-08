from functools import partial
from typing import Optional,Type

from langchain_core.callbacks.manager import (
    Callbacks,
)
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, format_document
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.retrievers import BaseRetriever

from langchain.tools import Tool,BaseTool

class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")

class ArticleInfoInput(BaseModel):
    """Input to the retriever."""

    title: str = Field(description="title of your paper")
    keyword: str = Field(description="the keywords of your paper")
    abstract: str = Field(description="the abstract of your paper")
    citation: str = Field(description="the citations")
def _get_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks = None,
) -> str:
    docs = retriever.get_relevant_documents(query, callbacks=callbacks)
    

    return document_separator.join(
        format_document(doc, document_prompt) for doc in docs
    )


async def _aget_relevant_documents(
    query: str,
    retriever: BaseRetriever,
    document_prompt: BasePromptTemplate,
    document_separator: str,
    callbacks: Callbacks = None,
) -> str:
    docs = await retriever.aget_relevant_documents(query, callbacks=callbacks)
    return document_separator.join(
        format_document(doc, document_prompt) for doc in docs
    )


def create_retriever_article_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
) -> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    document_prompt = document_prompt or PromptTemplate.from_template("""
Title: {title}
Content: {page_content}""")
    
    func = partial(
        _get_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
    )
    afunc = partial(
        _aget_relevant_documents,
        retriever=retriever,
        document_prompt=document_prompt,
        document_separator=document_separator,
    )
    return Tool(
        name=name,
        description=description,
        func=func,
        coroutine=afunc,
        args_schema=RetrieverInput,
    )


from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class ArticleFilterTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = ArticleInfoInput
    retriver: BaseRetriever = None

    def _run(
        self, *args, **kwargs, 
    ) -> str:
        """Use the tool."""
        return "test"

    async def _arun(
        self, *args, **kwargs, 
    ) -> str:
        """Use the tool asynchronously."""
        return "test"


def create_filter_article_tool(name ="filter_article",
                               description: str = "filter article from DataBase",
                               retriver = None):
    # func = partial(
    #     _filter_article_names
    # )
    
    return ArticleFilterTool(
        name=name,
        description=description,
        retriver = retriver,
        args_schema=ArticleInfoInput
    )