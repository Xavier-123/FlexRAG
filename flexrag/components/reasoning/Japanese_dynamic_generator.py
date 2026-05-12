"""
vLLM-compatible structured output generator with Two-Step Strategy Logic & Dynamic Strategy Filtering.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from flexrag.components.reasoning.base import BaseGenerator
from flexrag.common.schema import GenOutput

logger = logging.getLogger(__name__)

# ==========================================
# 策略知识库定义 (Strategy Repository)
# ==========================================
STRATEGY_REPO: Dict[str, str] = {
    "拒绝回答": "【策略：拒绝回答】适用：问题涉及机密信息、超范围、合规风险、知识库不足。要求：礼貌拒绝。",
    "用户主动终止咨询": "【策略：用户主动终止咨询】适用：意图状态放弃或用户明确结束。要求：确认终止并礼貌询问其他需求。",
    "确认/追问": "【策略：确认/追问】适用：意图非完成/放弃，且用户关键信息严重缺失。要求：精准提问，严禁强行追问。",
    "直接作答": "【策略：直接作答】适用：问题清晰，且知识库存在高度匹配内容。要求：结构化、分步完整输出。",
    "重复说明": "【策略：重复说明】适用：用户重复提问、无法理解。要求：更换表述方式，重新解读。",
    "问候/话题过渡": "【策略：问候/话题过渡】适用：问候、不满、闲聊等。要求：礼貌回应，平稳过渡至业务。",
    "问题已解决": "【策略：问题已解决】适用：意图状态处理完成，或明确解决。要求：礼貌确认，平稳结束对话。"
}

# ==========================================
# 静态 Schema 定义（修复 vLLM 崩溃的关键）
# ==========================================
class StrategyOutput(BaseModel):
    """制定对话系统的应答策略"""  # 类注释必须保留，LangChain 会将其作为 Tool 的描述
    first_layer_strategy: str = Field(description="第一层：宏观战略决策，必须严格从系统提示词的候选策略中选择一项")
    second_layer_strategy: str = Field(description="第二层：微观策略细化描述")
    selected_candidate_id: List[str] = Field(default_factory=list, description="可解答当前问题的全部高度关联知识ID，无则填空数组")
    reasoning: str = Field(description="决策理由：1.用户意图分析；2.知识筛选逻辑；3.策略匹配依据")
    suggested_question: str = Field(description="仅策略为「确认/追问」时填写，按优先级仅写单个问题，无则留空")
    additional_guidance: str = Field(description="其他补充提示信息，无则留空")


# ==========================================
# 动态 Prompt 模板构建
# ==========================================
def build_strategy_prompt(selected_strategies: List[str]) -> str:
    """根据过滤出的策略，动态生成系统提示词"""

    strategy_names_str = "、".join(selected_strategies)
    strategy_details_str = "\n".join([STRATEGY_REPO[name] for name in selected_strategies])

    prompt = f'''1. 角色定义
你为银行智能呼叫中心战略决策引擎。核心任务为：依据用户会话状态、对话上下文、当前用户输入，制定最优应答策略。当前服务限定特定专业领域，自身原有知识均可能已过期，因此必须严格遵照提供的知识库内容进行决策，禁止凭借自有知识自行推断。

2. 输入数据结构
你将接收以下输入信息（包含在 Human Message 中）：会话状态、对话上下文、当前用户输入。

3. 核心原则：意图与知识匹配
为本决策最高优先级原则，必须严格遵守。
3.1 知识严格筛选
1. 参考知识库内容繁杂，需忽略与当前用户意图、用户输入完全无关，或前提条件与用户状态不匹配的内容；仅依据和用户意图高度匹配的知识制定决策。
2. 若现有知识库内容不足以解答用户问题、或即使追问也无法作答，直接选择拒绝回答，禁止选择「确认 / 追问」。
3. 仅可使用高度相关知识；若仅名词重合、无法直接解答用户问题，禁止强行拼凑内容作答，需选择拒绝回答。
4. 若知识高度相关，但概念、前提、适用背景完全不同，严禁随意套用作答。
5. 若知识高度相关，但内容无法对应用户诉求，禁止强行应答，需选择拒绝回答。
6. 禁止将意图识别结果作为回答内容、禁止依托意图信息推导结论；所有应答内容只能来源于参考知识库。
3.2 意图优先
所有策略选择必须建立在精准解读用户意图的基础上。意图模糊时，优先选择「确认 / 追问」；意图明确但知识库信息不足时，选择「拒绝回答」。
3.3 状态驱动决策
重点关注意图模块内的intent_status（意图状态）字段：
• 状态为「处理完成」→ 优先选用「问题已解决」策略
• 状态为「放弃办理」→ 优先选用「用户主动终止咨询」策略
• 状态为「处理中」且用户信息缺失 → 优先选用「确认 / 追问」策略

4. 决策框架
采用双层决策模型
4.1 第一层：宏观战略决策
必须从以下候选策略中选定唯一一项（严禁输出列表以外的策略）：
[{strategy_names_str}]

4.2 第二层：微观策略细化
在选定的宏观策略下，结合实际场景匹配对应的细化执行方案

5. 候选策略库详情 (当前场景已过滤)
{strategy_details_str}

6. 决策优先级与注意事项
1. 合规安全绝对优先。2. 知识关联性核验。3. 状态驱动收尾。4. 信息完整性。5. 上下文连贯性。6. 多问题处理。7. 应答边界管控。8. 策略判定优先级优先核验是否缺失信息。

7. 输出自检要求
输出前必须完成自检核验，确保：时效/资料/资质限制/网点信息一致，未造假信息。

8. 输出格式
请严格输出合法 JSON，不允许输出 markdown，不允许解释。
输出格式:
{{
  "first_layer_strategy": 第一层：宏观战略决策，必须严格从系统提示词的候选策略中选择一项,
  "second_layer_strategy": 第二层：微观策略细化描述,
  "selected_candidate_id": [可解答当前问题的全部高度关联知识ID，无则填空数组],
  "reasoning": 决策理由：1.用户意图分析；2.知识筛选逻辑；3.策略匹配依据,
  "suggested_question": 仅策略为「确认/追问」时填写，按优先级仅写单个问题，无则留空,
  "additional_guidance": 其他补充提示信息，无则留空,
}}
'''

    return prompt

# ==========================================
# 最终答案 Prompt (静态保持不变)
# ==========================================
_SYSTEM_PROMPT_FINAL_ANSWER = '''角色定义
你是一名专业的客户支持助手，需要根据对话历史、用户当前输入以及指定的应答策略，撰写友好且准确的回复。
请注意：
你仅为问答助手，不具备业务代办、事项整理、问题记录、查询用户个人信息等功能。
回复中禁止出现「代为办理手续」「为您整理」「为您查询」等超出自身职责的表述。
同时，你的立场仅限依照应答策略进行回复，禁止在回答中擅自断定、暗示或默认用户「已理解 / 已清楚 / 无疑问 / 问题已解决」。

应答要求
1. 策略契合：严格遵循给定的「应答策略」与参考资料，不得编造信息。
2. 语气：亲切自然、专业得体、礼貌柔和，避免生硬机械的措辞。向用户确认事项时需礼貌友好，可搭配「明白了。想跟您确认一下」这类表达。
3. 内容规则：
  • 若首层策略为拒绝回答：使用固定文案：「非常抱歉。关于您的问题，这边无法即时答复，将为您转接专属客服，请您稍作等待。」
  • 若首层策略为用户主动结束咨询：尊重用户意愿，以「请问您还有其他问题吗？」收尾。
  • 若首层策略为用户问题已解决：确认用户结束意向，礼貌收尾。
  • 若首层策略为确认 / 追问：以参考提问内容为基础，用友好礼貌的语气反问，以疑问句结尾。
  • 若首层策略为直接回答：完整解答后，询问用户是否还有其他疑问，以疑问句收尾。
  • 若首层策略为重复说明：重新解释内容后，询问用户是否还有其他疑问，以疑问句收尾。
  • 若首层策略为问候 / 话术过渡：自然礼貌地回应即可。
4. 字数：30～100 字（问候类可适当缩短）
5. 准确性：严格依据应答策略与参考资料作答，禁止使用自身知识库、捏造内容。
6. 特殊要求：除用户重复提问外，不得重复赘述已解答过的内容。严禁添加「需要我帮您办理手续吗」等提问。

输出要求：
请严格输出合法 JSON，不允许输出 markdown，不允许解释。
输出格式:
{{
  "answer": The final generated answer, 类型为 str
  "evidence": Source document excerpts used to produce the answer, 类型为 List[str]
}}
请提取最终的文本作为 'answer'，并将使用的相关知识片段作为 'evidence'。'''


class JapaneseOpenAIGenerator(BaseGenerator):
    """Answer generator that calls an LLM with structured output in two steps."""

    def __init__(
            self,
            model: str = "Qwen/Qwen2.5-7B-Instruct",
            api_key: str | None = None,
            base_url: str | None = None,
            temperature: float = 0.0,
    ) -> None:
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,  # type: ignore[arg-type]
            base_url=base_url,
            temperature=temperature,
            model_kwargs={"top_p": 0.85}  # 可以加一点限制防止生成幻觉字段
        )
        # 初始化时静态绑定 Schema，避免每次请求重建导致报错和性能损耗
        # self._strategy_chain = self.llm.with_structured_output(StrategyOutput)
        # self._final_answer_chain = self.llm.with_structured_output(GenOutput)
        self._strategy_chain = self.llm
        self._final_answer_chain = self.llm

    # ------------------------------------------------------------------
    # 前置策略过滤模块 (规则 / 小模型)
    # ------------------------------------------------------------------
    def _pre_filter_strategies(self, query: str, context: str, session_state: dict) -> List[str]:
        """动态缩小策略范围"""
        intent_status = session_state.get("intent_space", {}).get("intent_status", "")
        selected = set()

        if intent_status == "处理完成":
            selected.add("问题已解决")
            selected.add("问候/话题过渡")
        elif intent_status == "放弃办理":
            selected.add("用户主动终止咨询")
            selected.add("问候/话题过渡")

        greeting_patterns = r"^(你好|在吗|有人吗|谢谢|再见|拜拜|ok|好的|明白了|知道啦)$"
        if re.match(greeting_patterns, query.strip(), re.IGNORECASE):
            selected.add("问候/话题过渡")
            selected.add("用户主动终止咨询")
            selected.add("问题已解决")

        if not selected or intent_status == "处理中":
            selected.update(["拒绝回答", "直接作答", "确认/追问", "重复说明"])

            if not context or context.strip() == "":
                selected.add("问候/话题过渡")

        if not selected:
            return list(STRATEGY_REPO.keys())

        return list(selected)

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    async def generate(
            self,
            query: str,
            context: str,
            accumulated_context: list[str],
            source_documents: list[str],
    ) -> GenOutput:

        history_str = "\n".join(accumulated_context) if accumulated_context else "无"

        # 构造会话状态
        session_state = {
            "intent_space": {"intent_status": "处理中"},
            "product_service_space": context,
            "user_space": {}
        }

        # 1. 动态过滤策略并生成 Prompt
        candidate_strategies = self._pre_filter_strategies(query, context, session_state)
        logger.info(f"Pre-filtered candidate strategies: {candidate_strategies}")
        dynamic_system_prompt = build_strategy_prompt(candidate_strategies)

        # ==========================================
        # Step 1: 制定应答策略
        # ==========================================
        step1_human_prompt = (
            f"会话状态：\n{json.dumps(session_state, ensure_ascii=False)}\n\n"
            f"对话上下文：\n{history_str}\n\n"
            f"当前用户输入：\n{query}\n"
        )

        logger.info("Step 1: Generating Response Strategy (with Dynamic Prompt)...")
        try:
            # 直接使用在 __init__ 中初始化的静态 chain，避免 vLLM 崩溃
            response = await self._strategy_chain.ainvoke(
            # strategy_result: StrategyOutput = await self._strategy_chain.ainvoke( # type: ignore[assignment]
                [
                    SystemMessage(content=dynamic_system_prompt),
                    HumanMessage(content=step1_human_prompt),
                ]
            )

            content = response.content

            # 去除 markdown json
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            try:
                data = json.loads(content)
                strategy_result = StrategyOutput(**data)
            except Exception as e:
                logger.exception(f"Strategy parse failed: {content}")
                raise

            logger.info(f"Step 1 Strategy output: {strategy_result.model_dump_json(ensure_ascii=False)}")
        except Exception as e:
            logger.exception(f"OpenAI generator failed at Step 1 (Strategy). \n  {e}")
            raise

        # ==========================================
        # Step 2: 生成最终答案
        # ==========================================
        step2_human_prompt = (
            f"对话历史：\n{history_str}\n\n"
            f"用户当前输入：\n{query}\n\n"
            f"应答策略：\n{strategy_result.model_dump_json(ensure_ascii=False, indent=2)}\n\n"
            f"参考资料：\n{context}\n\n"
            "请根据上述信息生成符合策略和格式要求的最终回答。将最终文本填入 'answer'，引用的资料填入 'evidence'。"
        )

        logger.info("Step 2: Generating Final Answer based on Strategy...")
        try:
            # final_result: GenOutput = await self._final_answer_chain.ainvoke(  # type: ignore[assignment]
            response = await self._final_answer_chain.ainvoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT_FINAL_ANSWER),
                    HumanMessage(content=step2_human_prompt),
                ]
            )

            content = response.content

            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            try:
                data = json.loads(content)
                final_result = GenOutput(**data)
            except Exception as e:
                print(e)
                logger.exception(f"Final parse failed: {content}")
                raise

            logger.info(f"Step 2 LLM structured output response: {final_result}")
        except Exception as e:
            logger.exception(f"OpenAI generator failed at Step 2 (Final Answer). \n  {e}")
            raise

        # Safety net: 兜底逻辑
        if not final_result.evidence and source_documents:
            final_result = GenOutput(
                answer=final_result.answer,
                evidence=source_documents[:3]
            )

        logger.debug(f"Generated final answer: {final_result.answer}")
        return final_result