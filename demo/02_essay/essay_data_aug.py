
from typing import Dict, Any, Optional, List

from qianfan.dataset import Dataset
from qianfan.common import Prompt
from qianfan.utils.pydantic import BaseModel

with open("prompt/essay_generating.prompt", mode="r") as f:
    generating_prompt_template = f.read()

with open("prompt/essay_scoring.prompt", mode="r") as f:
    correction_prompt_template = f.read()

# 为了方便我们批量发起不同类型作文数据的生成任务
# 我们定义了一个专门的结构体来传参
class TaskUnit(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    essay_generating_rules: str = ""
    is_bad_essay_case_set: bool = False

# 并且在这里初始化这个任务参数对象列表
task_unit_list: List[TaskUnit] = [
    TaskUnit(name="great_essay", essay_generating_rules="特别注意：请认真阅读作文要求，对照作文批改要求，生成一篇在内容、表达和发展上都非常好的优秀作文，最后返回生成的完整作文。"),
    TaskUnit(name="good_essay", essay_generating_rules="特别注意：请认真阅读作文要求，对照作文批改要求，生成一篇在内容、表达和发展上有少量缺陷的普通作文，最后返回生成的完整作文。"),
    TaskUnit(name="normal_essay", essay_generating_rules="特别注意：请认真阅读作文要求，对照作文批改要求，生成一篇中心思想偏离题意、内容单薄、表达混乱、没有发展的低质量作文，最后返回生成的完整作文。"),
    TaskUnit(name="failed_essay", essay_generating_rules="特别注意：请认真阅读作文要求，对照作文批改要求，生成一个字数不超过400字的残篇，最后返回生成的残篇作文。", is_bad_essay_case_set=True),
]

# 生成作文
def generate_essay_dataset(unit: TaskUnit, title_dataset: Dataset, service_model_name: str = "ERNIE-4.0-8K") -> Dataset:
    prompt_template = Prompt(
        template=generating_prompt_template.format(unit.essay_generating_rules),
        identifier="{}",
    )

    # 生成作文
    content_dataset = title_dataset.test_using_llm(
        service_model=service_model_name,
        prompt_template=prompt_template,
        slice_size=50,
        does_show_latency=False,
        # 还可以根据自身需要设置批量请求时的 TPM / RPM 限制
        # request_per_minute=100,
        # token_per_minute=120000,
    )

    def _format(entry: Dict[str, Any]) ->  Dict[str, Any]:
        return {
            "title": entry["title"],
            "essay": entry["llm_output"],
        }

    # 格式变换
    content_dataset.map(_format)
    content_dataset.input_columns = ["title", "essay"]
        
    return content_dataset


# 生成评分
def generate_correction_dataset(content_dataset: Dataset, service_model_name: str = "ERNIE-4.0-8K") -> Dataset:
    # 接下来我们继续去生成作文的评分
    correction_result_ds = content_dataset.test_using_llm(
        service_model=service_model_name,
        prompt_template=Prompt(template=correction_prompt_template, identifier="{{}}"),
        request_per_minute=50,
        does_show_latency=False,
        slice_size=50,
    )
    return correction_result_ds
