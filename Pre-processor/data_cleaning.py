"""
Text preprocessing utilities for HuggingFace blog articles.
Extracts date and cleans article text by removing title and date header.
"""

import re
import json
from typing import Dict, Tuple, Optional

# Pattern to match the header: Title + "Published" + Date + optional "Update on GitHub"
# Example: "Our Transformers Code Agent beats the GAIA benchmark 🏅\n\nPublished July 1, 2024 Update on GitHub"
HEADER_PATTERN = re.compile(
    r'^(.*?)\n+Published\s+(.*?)(?:\s+Update on GitHub)?\n',
    re.DOTALL | re.MULTILINE
)

# Alternative pattern if title might be on same line
HEADER_PATTERN_ALT = re.compile(
    r'^(.*?)Published\s+(.*?)(?:\s+Update on GitHub)?\n',
    re.DOTALL | re.MULTILINE
)


def extract_date_from_text(text: str) -> Optional[str]:
    """
    Extract publication date from article text.
    
    Args:
        text: Raw article text containing "Published <date>" pattern
        
    Returns:
        Date string (e.g., "July 1, 2024") or None if not found
    """
    match = HEADER_PATTERN.search(text)
    if match:
        date_str = match.group(2).strip()
        return date_str
    
    # Try alternative pattern
    match = HEADER_PATTERN_ALT.search(text)
    if match:
        date_str = match.group(2).strip()
        return date_str
    
    return None



def remove_header_from_text(text: str) -> str:
    """
    Remove title and publication date header from article text.
    Keeps only the actual content after "Published <date> Update on GitHub".
    
    Args:
        text: Raw article text
        
    Returns:
        Cleaned text without header
    """
    match = HEADER_PATTERN.search(text)
    if match:
        # Return everything after the match
        cleaned = text[match.end():].strip()
        return cleaned
    
    # Try alternative pattern
    match = HEADER_PATTERN_ALT.search(text)
    if match:
        cleaned = text[match.end():].strip()
        return cleaned
    
    # If no match, return original (don't break existing data)
    return text


def preprocess_article(article: Dict) -> Dict:
    """
    Preprocess a single article: extract date and clean text.
    
    Args:
        article: Dict with at least "Text" field and "link"
        
    Returns:
        Updated article dict with "Date_Extracted", "Text_Original", and cleaned "Text"
    """
    text = article.get("text", "")
    
    if not text:
        return article
    
    # Extract date
    date_extracted = extract_date_from_text(text)
    if date_extracted:
        article["date"] = date_extracted
    
    # Store original text before cleaning (optional, for debugging)    
    # Clean the text
    article["text"] = remove_header_from_text(text)
    
    return article


def preprocess_dataset(input_file: str, output_file: str):
    """
    Preprocess entire dataset JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (can be same as input)
    """
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {input_file}")
    
    # Process each article
    processed_count = 0
    date_extracted_count = 0
    
    for article in articles:
        if isinstance(article, dict) and "Text" in article and not article.get("_error"):
            original_text = article["Text"]
            preprocess_article(article)
            
            if article["Text"] != original_text:
                processed_count += 1
            
            if article.get("Date_Extracted"):
                date_extracted_count += 1
    
    # Save processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {processed_count} articles")
    print(f"Extracted dates from {date_extracted_count} articles")
    print(f"Saved to {output_file}")


# Example usage and testing
if __name__ == "__main__":
    # Test with your sample
    sample_text = """Our Transformers Code Agent beats the GAIA benchmark 🏅

Published July 1, 2024 Update on GitHub

After some experiments, we were impressed by the performance of Transformers Agents to build agentic systems..."""
    
    print("=" * 60)
    print("TESTING EXTRACTION")
    print("=" * 60)
    
    article = {
        "Title": "Our Transformers Code Agent beats the GAIA benchmark \ud83c\udfc5",
        "Authors": [],
        "Publish Date": None,
        "text": "Our Transformers Code Agent beats the GAIA benchmark \ud83c\udfc5\n\nPublished July 1, 2024 Update on GitHub\n\nAfter some experiments, we were impressed by the performance of Transformers Agents to build agentic systems, so we wanted to see how good it was! We tested using a Code Agent built with the library on the GAIA benchmark, arguably the most difficult and comprehensive agent benchmark\u2026 and ended up on top!\n\nThe framework transformers.agents used in this blog post has now been upgraded to the stand-alone library smolagents! The two libraries have very similar APIs, so switching is easy. Go checkout the smolagents introduction blog here.\n\nGAIA: a tough benchmark for Agents\n\nWhat are agents?\n\nIn one sentence: an agent is any system based on an LLM that can call external tools or not, depending on the need for the current use case and iterate on further steps based on the LLM output. Tools can include anything from a Web search API to a Python interpreter.\n\nFor a visual analogy: all programs could be described as graphs. Do A, then do B. If/else switches are forks in the graph, but they do not change its structure. We define agents as the systems where the LLM outputs will change the structure of the graph. An agent decides to call tool A or tool B or nothing, it decides to run one more step or not: these change the structure of the graph. You could integrate an LLM in a fixed workflow, as in LLM judge, without it being an agent system, because the LLM output will not change the structure of the graph\n\nHere is an illustration for two different system that perform Retrieval Augmented Generation: one is the classical, its graph is fixed. But the other is agentic, one loop in the graph can be repeated as needed.\n\nAgent systems give LLMs superpowers. For more detail, read our earlier blog post on the release of Transformers Agents 2.0.\n\nGAIA is the most comprehensive benchmark for agents. The questions in GAIA are very difficult and highlight certain difficulties of LLM-based systems.\n\nHere is an example of a tricky question:\n\nWhich of the fruits shown in the 2008 painting \"Embroidery from Uzbekistan\" were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film \"The Last Voyage\"? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o'clock position. Use the plural form of each fruit.\n\nYou can see this question involves several difficulties:\n\nAnswering in a constrained format.\n\nMultimodal abilities to read the fruits from the image\n\nSeveral informations to gather, some depending on the others: The fruits on the picture The identity of the ocean liner used as a floating prop for \u201cThe Last Voyage\u201d The October 1949 breakfast menu for the above ocean liner\n\nThe above forces the correct solving trajectory to use several chained steps.\n\nSolving this requires both high-level planning abilities and rigorous execution, which are precisely two areas where LLMs struggle.\n\nTherefore, it\u2019s an excellent test set for agent systems!\n\nOn GAIA\u2019s public leaderboard, GPT-4-Turbo does not reach 7% on average. The top submission is (was) an Autogen-based solution with a complex multi-agent system that makes use of OpenAI\u2019s tool calling functions, it reaches 40%.\n\nLet\u2019s take them on. \ud83e\udd4a\n\nBuilding the right tools \ud83d\udee0\ufe0f\n\nWe used three main tools to solve GAIA questions:\n\na. Web browser\n\nFor web browsing, we mostly reused the Markdown web browser from Autogen team\u2019s submission. It comprises a Browser class storing the current browser state, and several tools for web navigation, like visit_page , page_down or find_in_page . This tool returns markdown representations of the current viewport. Using markdown compresses web pages information a lot, which could lead to some misses, compared to other solutions like taking a screenshot and using a vision model. However, we found that the tool was overall performing well without being too complex to use or edit.\n\nNote: we think that a good way to improve this tool in the future would be to to load pages using selenium package rather than requests. This would allow us to load javascript (many pages cannot load properly without javascript) and accepting cookies to access some pages.\n\nb. File inspector\n\nMany GAIA questions rely on attached files from a variety of type, such as .xls , .mp3 , .pdf , etc. These files need to be properly parsed.. Once again, we use Autogen\u2019s tool since it works really well.\n\nMany thanks to the Autogen team for open-sourcing their work. It sped up our development process by weeks to use these tools! \ud83e\udd17\n\nc. Code interpreter\n\nWe will have no need for this since our agent naturally generates and executes Python code: see more below.\n\nCode Agent \ud83e\uddd1\u200d\ud83d\udcbb\n\nWhy a Code Agent?\n\nAs shown by Wang et al. (2024), letting the agent express its actions in code has several advantages compared to using dictionary-like outputs such as JSON. For us, the main advantage is that code is a very optimized way to express complex sequences of actions. Arguably if there had been a better way to rigorously express detailed actions than our current programming languages, it would have become a new programming language!\n\nConsider this example given in their paper:\n\nIt highlights several advantages of using code:\n\nCode actions are much more concise than JSON. Need to run 4 parallel streams of 5 consecutive actions ? In JSON, you would need to generate 20 JSON blobs, each in their separate step; in Code it\u2019s only 1 step. On average, the paper shows that Code actions require 30% fewer steps than JSON, which amounts to an equivalent reduction in the tokens generated. Since LLM calls are often the dimensioning cost of agent systems, it means your agent system runs are ~30% cheaper.\n\nthan JSON. Code enables to re-use tools from common libraries\n\nUsing code gets better performance in benchmarks, due to two reasons: It\u2019s a more intuitive way to express actions LLMs have lots of code in their training data, which possibly makes them more fluent in code-writing than in JSON writing.\n\n\n\nWe confirmed these points during our experiments on agent_reasoning_benchmark.\n\nFrom our latest experiments of building transformers agents, we also observed additional advantages:\n\nIt is much easier to store an element as a named variable in code. For example, need to store this rock image generated by a tool for later use? No problem in code: using \u201crock_image = image_generation_tool(\u201cA picture of a rock\u201d)\u201d will store the variable under the key \u201crock_image\u201d in your dictionary of variables. Later the LLM can just use its value in any code blob by referring to it again as \u201crock_image\u201d. In JSON you would have to do some complicated gymnastics to create a name under which to store this image, so that the LLM later knows how to access it again. For instance, save any output of the image generation tool under \u201cimage_{i}.png\u201d, and trust that the LLM will later understand that image_4.png is the output of the tool call that precedes it in memory? Or let the LLM also output a \u201coutput_name\u201d key to choose under which name to store the variable, thus complicating the structure of your action JSON?\n\nAgent logs are considerably more readable.\n\nImplementation of Transformers Agents\u2019 CodeAgent\n\nThe thing with LLM generated code is that it can be really unsafe to execute as is. If you let an LLM write and execute code without guardrails, it could hallucinate anything: for instance that all your personal files need to be erased by copies of the Dune lore, or that this audio of you singing the Frozen theme needs to be shared on your blog!\n\nSo for our agents, we had to make code execution secure. The usual approach is top-down: \u201cuse a fully functional python interpreter, but forbid certain actions\u201d.\n\nTo be more safe, we preferred to go the opposite way, and build a LLM-safe Python interpreter from the ground-up. Given a Python code blob provided by the LLM, our interpreter starts from the Abstract Syntax Tree representation of the code given by the ast python module. It executes the tree nodes one by one, following the tree structure, and stops at any operation that was not explicitly authorised\n\nFor example, an import statement will first check if the import is explicitly mentioned in the user-defined list of authorized_imports : if not, it does not execute. We include a default list of built-in standard Python functions, comprising for instance print and range . Anything outside of it will not be executed except explicitly authorized by the user. For instance, open (as in with open(\"path.txt\", \"w\") as file: ) is not authorized.\n\nWhen encountering a function call ( ast.Call ), if the function name is one of the user-defined tools, the tool is called with the arguments to the call. If it\u2019s another function defined and allowed earlier, it gets run normally.\n\nWe also do several tweaks to help with LLM usage of the interpreter:\n\nWe cap the number of operations in execution to prevent infinite loops caused by issues in LLM-generated code: at each operation, a counter gets incremented, and if it reaches a certain threshold the execution is interrupted\n\nWe cap the number of lines in print outputs to avoid flooding the context length of the LLM with junk. For instance if the LLM reads a 1M lines text files and decides to print every line, at some point this output will be truncated, so that the agent memory does not explode.\n\nBasic multi-agent orchestration\n\nWeb browsing is a very context-rich activity, but most of the retrieved context is actually useless. For instance, in the above GAIA question, the only important information to get is the image of the painting \"Embroidery from Uzbekistan\". Anything around it, like the content of the blog we found it on, is generally useless for the broader task solving.\n\nTo solve this, using a multi-agent step makes sense! For example, we can create a manager agent and a web search agent. The manager agent should solve the higher-level task, and assign specific web search task to the web search agent. The web search agent should return only the useful outputs of its search, so that the manager is not cluttered with useless information.\n\nWe created exactly this multi-agent orchestration in our workflow:\n\nThe top level agent is a ReactCodeAgent. It natively handles code since its actions are formulated and executed in Python. It has access to these tools: file_inspector to read text files, with an optional question argument to not return the whole content of the file but only return its answer to the specific question based on the content visualizer to specifically answer questions about images. search_agent to browse the web. More specifically, this Tool is just a wrapper around a Web Search agent, which is a JSON agent (JSON still works well for strictly sequential tasks, like web browsing where you scroll down, then navigate to a new page, and so on). This agent in turn has access to the web browsing tools: informational_web_search page_down find_in_page \u2026 (full list at this line)\n\n\n\nThis embedding of an agent as a tool is a naive way to do multi-agent orchestration, but we wanted to see how far we could push it - and it turns out that it goes quite far!\n\nPlanning component \ud83d\uddfa\ufe0f\n\nThere is now an entire zoo of planning strategies, so we opted for a relatively simple plan-ahead workflow. Every N steps we generate two things:\n\na summary of facts we know or we can derive from context and facts we need to discover\n\na step-by-step plan of how to solve the task given fresh observations and the factual summary above\n\nThe parameter N can be tuned for better performance on the target use cas: we chose N=2 for the manager agent and N=5 for the web search agent.\n\nAn interesting discovery was that if we do not provide the previous version of the plan as input, the score goes up. An intuitive explanation is that it\u2019s common for LLMs to be strongly biased towards any relevant information available in the context. If the previous version of the plan is present in the prompt, an LLM is likely to heavily reuse it instead of re-evaluating the approach and re-generating a plan when needed.\n\nBoth the summary of facts and the plan are then used as additional context to generate the next action. Planning encourages an LLM to choose a better trajectory by having all the steps to achieve the goal and the current state of affairs in front of it.\n\nResults \ud83c\udfc5\n\nHere is the final code used for our submission.\n\nWe get 44.2% on the validation set: so that means Transformers Agent\u2019s ReactCodeAgent is now #1 overall, with 4 points above the second! On the test set, we get 33.3%, so we rank #2, in front of Microsoft Autogen\u2019s submission, and we get the best average score on the hardcore Level 3 questions.\n\nThis is a data point to support that Code actions work better. Given their efficiency, we think Code actions will soon replace JSON/OAI format as the standard for agents writing their actions.\n\nLangChain and LlamaIndex do not support Code actions out of the box to our knowledge, Microsoft's Autogen has some support for Code actions (executing code in docker containers) but it looks like an annex to JSON actions. So Transformers Agents is the only library to make this format central!\n\nNext steps\n\nWe hope you enjoyed reading this blog post! And the work is just getting started, as we\u2019ll keep improving Transformers Agents, along several axes:\n\nLLM engine: Our submission was done with GPT-4o (alas), without any fine-tuning . Our hypothesis is that using a fine-tuned OS model would allow us to get rid of parsing errors, and score a bit higher!\n\nOur submission was done with GPT-4o (alas), . Our hypothesis is that using a fine-tuned OS model would allow us to get rid of parsing errors, and score a bit higher! Multi-agent orchestration: our is a naive one, with more seamless orchestration we could probably go a long way!\n\nour is a naive one, with more seamless orchestration we could probably go a long way! Web browser tool: using the selenium package, we could have a web browser that passes cookie banners and loads javascript, thus allowing us to read many pages that are for now not accessible.\n\nusing the package, we could have a web browser that passes cookie banners and loads javascript, thus allowing us to read many pages that are for now not accessible. Improve planning further: We\u2019re running some ablation tests with other options from the literature to see which method works best. We are planning to give a try to alternative implementations of existing components and also some new components. We will publish our updates when we have more insights!\n\nKeep an eye on Transformers Agents in the next few months! \ud83d\ude80\n\nAnd don\u2019t hesitate to reach out to us with your use cases, now that we have built internal expertise on Agents we\u2019ll be happy to lend a hand! \ud83e\udd1d",
        "link": "https://huggingface.co/blog/beating-gaia",
        "Summary": "Our Transformers Code Agent beats the GAIA benchmark \ud83c\udfc5Published July 1, 2024 Update on GitHubAfter some experiments, we were impressed by the performance of Transformers Agents to build agentic systems, so we wanted to see how good it was!\nWe tested using a Code Agent built with the library on the GAIA benchmark, arguably the most difficult and comprehensive agent benchmark\u2026 and ended up on top!\nCode Agent \ud83e\uddd1\u200d\ud83d\udcbbWhy a Code Agent?\nGiven their efficiency, we think Code actions will soon replace JSON/OAI format as the standard for agents writing their actions.\nLangChain and LlamaIndex do not support Code actions out of the box to our knowledge, Microsoft's Autogen has some support for Code actions (executing code in docker containers) but it looks like an annex to JSON actions.",
        "Keywords": [
            "benchmark",
            "beats",
            "gaia",
            "agent",
            "web",
            "transformers",
            "tools",
            "using",
            "agents",
            "llm",
            "actions",
            "code",
            "tool",
            "json"
        ]
    }
    # Extract date
    data = preprocess_article(article)

    data    
    # Or process in-place:
    # preprocess_dataset("hg_blogs_data.json", "hg_blogs_data.json")