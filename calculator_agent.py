import json
import ast
import operator
from groq import Groq
from groq import BadRequestError

client = Groq()
MODEL = "openai/gpt-oss-120b"


# ============================================================================
# Tool Implementations
# ============================================================================

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        operand = _safe_eval(node.operand)
        return ALLOWED_OPERATORS[type(node.op)](operand)
    raise ValueError("Unsupported expression.")


def calculate(expression: str) -> str:
    """Evaluate a basic mathematical expression safely."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def calculate_compound_interest(
    principal: float, rate: float, time: float, compounds_per_year: int = 12
) -> str:
    """Calculate compound interest on an investment."""
    amount = principal * (1 + rate / compounds_per_year) ** (compounds_per_year * time)
    interest = amount - principal
    return json.dumps({
        "principal": principal,
        "total_amount": round(amount, 2),
        "interest_earned": round(interest, 2),
    })


def calculate_percentage(number: float, percentage: float) -> str:
    """Calculate what percentage of a number equals."""
    result = (percentage / 100) * number
    return json.dumps({"result": round(result, 2)})


# Function registry
available_functions = {
    "calculate": calculate,
    "calculate_compound_interest": calculate_compound_interest,
    "calculate_percentage": calculate_percentage,
}


# ============================================================================
# Tool Schemas
# ============================================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression like '25 * 4 + 10' or '(100 - 50) / 2'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_compound_interest",
            "description": "Calculate compound interest on an investment",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "number",
                        "description": "The initial investment amount",
                    },
                    "rate": {
                        "type": "number",
                        "description": "The annual interest rate as a decimal (e.g., 0.05 for 5%)",
                    },
                    "time": {
                        "type": "number",
                        "description": "The time period in years",
                    },
                    "compounds_per_year": {
                        "type": "integer",
                        "description": "Number of times interest compounds per year (default: 12)",
                        "default": 12,
                    },
                },
                "required": ["principal", "rate", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_percentage",
            "description": "Calculate what a percentage of a number equals",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "number",
                        "description": "The base number",
                    },
                    "percentage": {
                        "type": "number",
                        "description": "The percentage to calculate",
                    },
                },
                "required": ["number", "percentage"],
            },
        },
    },
]


# ============================================================================
# Agentic Loop
# ============================================================================

def create_completion(messages: list[dict], allow_tools: bool = True):
    """Create a chat completion and recover from malformed tool calls."""
    request_kwargs = {"model": MODEL, "messages": messages}
    if allow_tools:
        request_kwargs.update({"tools": tools, "tool_choice": "auto"})

    try:
        return client.chat.completions.create(**request_kwargs)
    except BadRequestError as exc:
        error_text = str(exc)
        if "attempted to call tool" in error_text and allow_tools:
            fallback_messages = messages + [{
                "role": "system",
                "content": (
                    "Do not call tools. Use the previous tool outputs in the conversation "
                    "to provide the final answer directly."
                ),
            }]
            return client.chat.completions.create(
                model=MODEL,
                messages=fallback_messages,
                tool_choice="none",
            )
        raise


def run_agent(user_query: str, max_iterations: int = 10) -> str:
    """Run the calculator agent with an agentic tool-calling loop."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial calculator assistant. "
                "Use the provided tools to help with calculations."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    print(f"User: {user_query}\n")

    response = create_completion(messages, allow_tools=True)

    iteration = 0

    while response.choices[0].message.tool_calls and iteration < max_iterations:
        iteration += 1
        messages.append(response.choices[0].message)

        tool_calls = response.choices[0].message.tool_calls
        print(f"Iteration {iteration}: Model called {len(tool_calls)} tool(s)")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"  → {function_name}({function_args})")

            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            print(f"    ← {function_response}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_response,
            })

        response = create_completion(messages, allow_tools=True)
        print()

    final_answer = response.choices[0].message.content
    print(f"Assistant: {final_answer}")
    return final_answer


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    user_query = (
        "I'm investing $10,000 at 5% annual interest for 10 years, "
        "compounded monthly. After 10 years, I want to withdraw 25% "
        "for a down payment. How much will my down payment be, and "
        "how much will remain invested?"
    )
    run_agent(user_query)
    