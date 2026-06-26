from langgraph.graph import (
    StateGraph,
    START,
    END
)


from multi.state import State
from multi.agents import (
    content_strategist,
    communicator
)


builder=StateGraph(State)



def strategist_node(state):

    return content_strategist(
        state,
        state["thread_id"]
    )



builder.add_node(
    "strategist",
    strategist_node
)


builder.add_node(
    "communicator",
    communicator
)



builder.add_edge(
    START,
    "strategist"
)


builder.add_edge(
    "strategist",
    "communicator"
)


builder.add_edge(
    "communicator",
    END
)



graph=builder.compile()
