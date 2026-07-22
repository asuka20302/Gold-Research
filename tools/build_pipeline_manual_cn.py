from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "docs" / "黄金量化预测项目完整流程.docx"

FONT_CN = "Microsoft YaHei"
FONT_EN = "Calibri"
FONT_MONO = "Consolas"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "0B2545"
MUTED = "667085"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
CALLOUT = "F4F6F9"
WHITE = "FFFFFF"
GOLD = "7A5A00"
RED = "9B1C1C"


def set_run_font(
    run,
    name: str = FONT_CN,
    size: float | None = None,
    color: str | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT_EN)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT_EN)
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int], indent_dxa: int = 120):
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            width = widths_dxa[min(index, len(widths_dxa) - 1)]
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def style_table(table, header: bool = True):
    table.style = "Table Grid"
    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            if header and row_index == 0:
                set_cell_shading(cell, LIGHT_BLUE)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.line_spacing = 1.15
                for run in paragraph.runs:
                    set_run_font(
                        run,
                        size=9.2,
                        bold=(header and row_index == 0),
                        color=INK,
                    )


def add_table(doc, headers: list[str], rows: list[tuple], widths_dxa: list[int]):
    table = doc.add_table(rows=1, cols=len(headers))
    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = str(header)
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = str(value)
    set_table_geometry(table, widths_dxa)
    style_table(table)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(4)
    return table


def add_para(doc, text: str = "", bold=False, italic=False, color=None, align=None, after=6):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.25
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(text)
    set_run_font(run, size=11, color=color or "202124", bold=bold, italic=italic)
    return paragraph


def add_bullet(doc, text: str, level: int = 0):
    paragraph = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.25
    paragraph.paragraph_format.left_indent = Inches(0.375 + 0.25 * level)
    paragraph.paragraph_format.first_line_indent = Inches(-0.188)
    run = paragraph.add_run(text)
    set_run_font(run, size=11)
    return paragraph


def add_number(doc, text: str):
    paragraph = doc.add_paragraph(style="List Number")
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.25
    paragraph.paragraph_format.left_indent = Inches(0.375)
    paragraph.paragraph_format.first_line_indent = Inches(-0.188)
    run = paragraph.add_run(text)
    set_run_font(run, size=11)
    return paragraph


def add_code(doc, lines: list[str] | str):
    if isinstance(lines, list):
        text = "\n".join(lines)
    else:
        text = lines
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    set_cell_shading(cell, LIGHT_GRAY)
    set_table_geometry(table, [9360])
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.05
    run = paragraph.add_run(text)
    set_run_font(run, name=FONT_MONO, size=9.2, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_callout(doc, title: str, text: str, fill: str = CALLOUT, color: str = INK):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title)
    set_run_font(r, size=10.5, color=color, bold=True)
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(2)
    p2.paragraph_format.line_spacing = 1.2
    r2 = p2.add_run(text)
    set_run_font(r2, size=10.3, color="344054")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_heading(doc, text: str, level: int):
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    return paragraph


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("第 ")
    set_run_font(run, size=9, color=MUTED)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, end])
    run2 = paragraph.add_run(" 页")
    set_run_font(run2, size=9, color=MUTED)


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT_CN
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    normal.font.size = Pt(11)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    tokens = {
        1: (16, BLUE, 18, 10),
        2: (13, BLUE, 14, 7),
        3: (12, DARK_BLUE, 10, 5),
    }
    for level, (size, color, before, after) in tokens.items():
        style = styles[f"Heading {level}"]
        style.font.name = FONT_CN
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for name in ["List Bullet", "List Bullet 2", "List Number"]:
        style = styles[name]
        style.font.name = FONT_CN
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
        style.font.size = Pt(11)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.25


def configure_section(section):
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run("黄金量化预测项目  |  完整流程说明")
    set_run_font(r, size=9, color=MUTED, bold=True)

    footer = section.footer
    p2 = footer.paragraphs[0]
    add_page_number(p2)


def build_document():
    doc = Document()
    configure_styles(doc)
    configure_section(doc.sections[0])

    # Editorial-cover pattern.
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(92)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("研究流程手册")
    set_run_font(r, size=11, color=GOLD, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("黄金量化预测项目完整流程")
    set_run_font(r, size=28, color=INK, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(24)
    r = p.add_run("从数据获取、线性建模到涨跌金额误差与交易评估")
    set_run_font(r, size=14, color=DARK_BLUE)

    add_callout(
        doc,
        "当前状态",
        "研究原型。已建立可复现的数据、建模、PCA、稳健损失、Bagging、加权平均和金额误差分析流程；尚未达到生产部署标准。",
        fill=LIGHT_BLUE,
    )
    add_para(doc, f"版本日期：{date.today().isoformat()}", bold=True, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
    add_para(doc, "工作目录：GitHub 项目根目录", color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()

    add_heading(doc, "使用说明", 1)
    add_para(doc, "本手册记录当前项目的完整研究流程、模型决策、主要结果、已知限制及下一阶段计划。它用于项目汇报、代码复现和后续交接。")
    add_callout(
        doc,
        "重要口径",
        "高级 PCA、损失函数和加权模型不再使用方向准确率进行选择。核心目标是预测实际涨跌幅金额，并衡量预测值与真实值在百分点和人民币/克上的差异。",
        fill="FFF7E6",
        color=GOLD,
    )

    add_heading(doc, "流程目录", 2)
    for item in [
        "项目目标与流程总览",
        "数据收集、缓存与时间对齐",
        "预测目标与交易时间规则",
        "特征工程、市场状态与异常事件",
        "训练集、验证集与测试集",
        "预处理 Pipeline 与统计特征选择",
        "五类线性回归模型",
        "Bagging、PCA 与损失函数",
        "样本外加权平均",
        "涨跌金额和价格变化误差",
        "当前主要问题与下一阶段计划",
        "已完成的十分组 Huber 实验",
        "关键代码与输出文件",
    ]:
        add_number(doc, item)

    add_heading(doc, "1. 项目目标与流程总览", 1)
    add_para(doc, "项目目标是利用交易前已经获得的黄金、汇率、利率、通胀、原油和期货市场信息，预测 Au99.99 下一个交易时段从开盘到收盘的具体涨跌幅和价格变化。")
    add_table(
        doc,
        ["阶段", "输入", "处理", "输出"],
        [
            ("1. 数据", "Au99.99 与外部因子", "下载、缓存、日期对齐", "统一日频数据"),
            ("2. 特征", "历史价格和宏观数据", "收益率、动量、波动率、溢价、交互项", "候选解释变量"),
            ("3. 建模", "训练集", "AIC、五类回归、PCA、稳健损失", "候选预测模型"),
            ("4. 组合", "样本外预测", "Bagging 与误差加权", "组合预测"),
            ("5. 评估", "预测值与真实值", "百分点误差、人民币/克误差", "金额误差账本"),
            ("6. 交易", "最终预测", "仓位、成本与持有期规则", "P&L 与风险指标"),
        ],
        [1200, 2300, 3360, 2500],
    )

    add_heading(doc, "2. 数据收集与缓存", 1)
    add_para(doc, "主要研究品种为上海黄金交易所 Au99.99。程序优先读取本地缓存，只有在缓存不存在或明确要求刷新时才重新下载。")
    for text in [
        "国际现货黄金 XAU/USD。",
        "USD/CNH 和美元相关数据。",
        "原油价格与动量。",
        "美国名义利率、实际利率和盈亏平衡通胀率。",
        "VIX 与 GVZ 波动率。",
        "上期所黄金价格、结算价、成交量和持仓量。",
        "上海黄金相对于国际黄金的溢价。",
    ]:
        add_bullet(doc, text)
    add_code(doc, ["数据缓存目录：data/", "外部因子缓存：data/factors/", "主要 Au99.99 文件：data/Au99_99_20150101_20260703.csv"])

    add_heading(doc, "3. 日期对齐与信息可用时间", 1)
    add_para(doc, "全部外部数据与 Au99.99 交易日历对齐。海外市场因子根据其实际发布时间进行平移，尽量保证模型只使用预测时点之前可以获得的信息。")
    add_callout(doc, "防止未来信息泄漏", "不能因为两个市场使用同一个日历日期，就假设它们的收盘数据在中国市场开盘前已经可用。所有因子都必须按照真实信息时间检查。")

    add_heading(doc, "4. 预测目标与交易时间", 1)
    add_table(
        doc,
        ["项目", "当前定义"],
        [
            ("信号", "使用入场之前已经获得的信息计算"),
            ("入场", "下一个交易日开盘"),
            ("离场", "同一个交易日收盘"),
            ("预测期限", "一个交易时段"),
            ("主要标签", "开盘到收盘的对数收益率"),
        ],
        [2700, 6660],
    )
    add_code(doc, ["target_log_return = log(exit_close / entry_open)", "growth_pct = 100 * (exp(target_log_return) - 1)"])
    add_para(doc, "模型内部使用对数收益率以便进行回归；最终报告使用普通百分比涨跌幅和人民币/克价格变化，便于解释和交易评估。")

    add_heading(doc, "5. 特征工程", 1)
    add_para(doc, "原始市场数据被转换为能够描述趋势、波动、相对定价和宏观环境的特征。")
    add_table(
        doc,
        ["特征类别", "代表性特征", "经济含义"],
        [
            ("黄金自身", "一日收益、五日/二十日动量、均线偏离", "短期惯性和中期趋势"),
            ("波动率", "二十日波动率", "风险状态与预期波动幅度"),
            ("国际黄金", "XAU/USD 收益和动量", "国际定价传导"),
            ("汇率", "USD/CNH 收益", "人民币计价转换"),
            ("通胀与利率", "盈亏平衡通胀率、实际利率", "持有黄金的宏观机会成本"),
            ("商品", "原油动量", "通胀和风险环境代理"),
            ("跨市场", "上期所结算价收益、上海溢价", "境内外价差与本地定价"),
        ],
        [1700, 3100, 4560],
    )
    add_para(doc, "当前保留的重要交互项为：五日动量 × 二十日波动率。它允许动量影响随波动率变化。")

    add_heading(doc, "6. 市场状态与虚拟变量", 1)
    add_table(
        doc,
        ["标签", "市场状态", "处理方式"],
        [
            ("0", "正常市场", "基准组，全部状态虚拟变量为 0"),
            ("1", "高波动", "对应虚拟变量为 1"),
            ("2", "强上涨趋势", "对应虚拟变量为 1"),
            ("3", "强下跌趋势", "对应虚拟变量为 1"),
            ("4", "流动性压力代理", "对应虚拟变量为 1"),
        ],
        [1100, 2900, 5360],
    )
    add_para(doc, "状态虚拟变量和状态交互项已经进行验证测试，但没有带来足够稳定的改善。因此它们保留为诊断信息，不强制进入当前模型。")

    add_heading(doc, "7. 异常事件处理", 1)
    add_para(doc, "当前设置排除了七个异常事件窗口，共影响 30 条模型观测。这些时期涉及贸易政策、地缘政治、利率和流动性冲击。")
    add_callout(
        doc,
        "方法论限制",
        "这些事件窗口是在早期检查测试集残差之后确定的。因此过滤后的测试结果只能作为诊断，不能被宣传为完全未使用的独立样本证据。最终模型必须保留新的未来测试区间。",
        fill="FDECEC",
        color=RED,
    )

    add_heading(doc, "8. 训练集、验证集与测试集", 1)
    add_table(
        doc,
        ["数据部分", "观测数", "日期", "用途"],
        [
            ("训练集", "1,929", "2015-02-02 至 2023-01-05", "估计系数、PCA 和预处理参数"),
            ("验证集", "408", "2023-01-17 至 2024-09-20", "选择特征、模型、损失与组合方式"),
            ("测试集", "384", "2024-10-09 至 2026-06-23", "模型冻结后的诊断评估"),
        ],
        [1500, 1200, 2900, 3760],
    )
    add_para(doc, "主流程按时间顺序划分，不随机打乱交易日期。时间顺序是避免金融预测未来信息泄漏的核心要求。")

    add_heading(doc, "9. 数据预处理 Pipeline", 1)
    add_code(doc, ["Pipeline([", "    ('imputer', SimpleImputer(strategy='median')),", "    ('scaler', StandardScaler()),", "    ('model', regression_model),", "]) "])
    for text in [
        "缺失值只使用训练样本的中位数填补。",
        "标准化参数只使用训练样本计算。",
        "模型只接收经过相同预处理的特征。",
        "验证集和测试集不能反向影响预处理参数。",
    ]:
        add_bullet(doc, text)

    add_heading(doc, "10. 统计特征选择", 1)
    add_table(
        doc,
        ["方法", "检验对象", "项目中的用途"],
        [
            ("T 检验", "单个系数是否显著偏离 0", "识别单个弱变量"),
            ("ANOVA/部分 F", "一组系数是否可以同时视为 0", "按经济分组检查因子"),
            ("AIC", "拟合质量与参数数量之间的平衡", "当前生产特征选择方法"),
            ("BIC", "对复杂度施加更强惩罚", "较保守的对照模型"),
        ],
        [1700, 3100, 4560],
    )
    add_para(doc, "当前使用带 P 值约束的 AIC 后向剔除：只有 p>0.05 的变量可以进入删除候选，并且删除后 AIC 必须改善。保留交互项时必须同时保留它的基础变量。")
    add_callout(doc, "当前结果", "一日预测模型最终保留 14 个 AIC 特征。")

    add_heading(doc, "11. 五类回归模型", 1)
    add_table(
        doc,
        ["模型", "核心思想", "主要作用"],
        [
            ("OLS", "最小化平方误差", "基础线性模型和统计解释"),
            ("Ridge", "L2 系数惩罚", "处理共线性并稳定系数"),
            ("Lasso", "L1 系数惩罚", "压缩并可能删除变量"),
            ("Elastic Net", "L1 与 L2 结合", "在稀疏性和稳定性之间折中"),
            ("Huber", "普通误差平方损失，大误差近似线性损失", "降低极端观测影响"),
        ],
        [1700, 4000, 3660],
    )
    add_para(doc, "在原始五个模型中，Huber 通常表现最好，说明稳健损失对黄金极端波动具有一定价值。")

    add_heading(doc, "12. 移动区块 Bagging", 1)
    add_para(doc, "金融时间序列中的相邻交易日可能存在依赖关系，因此项目不是随机抽取单日，而是有放回地抽取连续数据区块。")
    for text in [
        "随机选择一个连续区块的起点。",
        "抽取连续 20 条训练观测。",
        "重复抽取直到形成一个与原训练集同等长度的 Bootstrap 样本。",
        "对每个样本拟合一个 Huber 模型。",
        "对 50 个模型的预测取算术平均。",
    ]:
        add_number(doc, text)
    add_code(doc, ["block_size = 20", "n_estimators = 50", "random_state = 42"])
    add_para(doc, "对 OLS、Ridge、Lasso 和 Elastic Net 进行 Bagging 后，验证改善没有稳定延续到测试阶段。目前只保留 Bagged Huber 作为基准。")

    add_heading(doc, "13. 主成分分析 PCA", 1)
    add_para(doc, "PCA 在中位数填补和标准化之后拟合，且只能使用训练样本。测试的主成分数量为 3、5、7、9、11、13 和 14。")
    add_table(
        doc,
        ["模型", "PCA 决策", "说明"],
        [
            ("OLS/Ridge/Lasso/Elastic Net", "拒绝", "金额误差没有同时改善"),
            ("Huber", "选择 11 个主成分", "解释约 97.8% 的标准化特征方差"),
        ],
        [3000, 2400, 3960],
    )
    add_para(doc, "PCA 单独改善较小，并会降低经济解释性；它与绝对误差损失组合时更有价值。")

    add_heading(doc, "14. 损失函数比较", 1)
    add_table(
        doc,
        ["损失函数", "优化重点", "结果"],
        [
            ("平方误差", "大误差受到平方惩罚", "OLS 基准"),
            ("Huber", "普通误差平方、大误差线性", "原五模型中最好"),
            ("绝对误差", "最小化绝对偏差，估计条件中位数", "PCA11 组合成为最强金额预测挑战者"),
        ],
        [2200, 3200, 3960],
    )
    add_code(doc, ["模型：QuantileRegressor", "quantile = 0.50", "alpha = 0", "PCA components = 11"])

    add_heading(doc, "15. 样本外加权平均", 1)
    add_para(doc, "权重使用五个扩展窗口的训练集样本外预测计算，并设置一个观测期的时间间隔。验证集只负责选择权重方法，不直接计算数值权重。")
    for text in [
        "等权重。",
        "反增长 MSE 权重。",
        "反增长 MAE 权重。",
        "非负且和为 1 的最优增长 MSE 权重。",
        "带稳定惩罚的最优权重。",
    ]:
        add_bullet(doc, text)
    add_table(
        doc,
        ["组合成员", "当前权重", "作用"],
        [
            ("移动区块 Bagged Huber", "39.62%", "保留原始特征上的稳健相关信息"),
            ("PCA11 绝对误差回归", "60.38%", "降低典型金额误差"),
        ],
        [3600, 1800, 3960],
    )
    add_para(doc, "完整五模型加权没有明显优势，因为 OLS、Ridge、Lasso 和 Elastic Net 的预测高度相似，缺乏组合所需的模型多样性。")

    add_heading(doc, "16. 涨跌金额与价格变化误差", 1)
    add_para(doc, "高级模型不再使用方向准确率。每个日期都直接比较预测涨跌金额和真实涨跌金额。")
    add_code(doc, ["actual_growth_pct = 100 * (exp(actual_log_return) - 1)", "predicted_growth_pct = 100 * (exp(predicted_log_return) - 1)", "growth_error_pp = predicted_growth_pct - actual_growth_pct", "price_change_error = predicted_exit_price - actual_exit_price"])
    add_table(
        doc,
        ["指标", "计算含义", "解释"],
        [
            ("增长 MAE", "平均绝对百分点误差", "典型交易日的平均预测偏差"),
            ("增长 RMSE", "百分点误差平方后平均再开根号", "更重视少数大错误"),
            ("波动幅度 MAE", "预测绝对幅度与实际绝对幅度之差", "判断是否低估或高估波动"),
            ("人民币/克变化 MAE", "预测价格变化与实际价格变化的平均绝对差", "直接连接交易金额"),
        ],
        [2200, 3500, 3660],
    )

    add_heading(doc, "17. 当前金额预测结果", 1)
    add_table(
        doc,
        ["诊断测试候选", "增长 MAE", "增长 RMSE", "幅度 MAE", "价格变化 MAE"],
        [
            ("Bagged Huber", "0.684 pp", "0.955 pp", "0.578 pp", "5.914 元/克"),
            ("PCA11 绝对误差", "0.669 pp", "0.921 pp", "0.573 pp", "5.768 元/克"),
            ("稳健双模型加权", "0.671 pp", "0.929 pp", "0.572 pp", "5.778 元/克"),
        ],
        [2600, 1500, 1500, 1500, 2260],
    )
    add_para(doc, "按照金额误差，PCA11 绝对误差模型目前最有潜力；加权模型只在波动幅度 MAE 上略好，因此还不能明确替代单独模型。")

    add_heading(doc, "18. 当前主要问题", 1)
    add_table(
        doc,
        ["实际波动区间", "平均实际幅度", "平均预测幅度", "低估比例"],
        [
            ("低于 0.5%", "0.235%", "0.235%", "54.6%"),
            ("0.5%–1%", "0.739%", "0.248%", "97.2%"),
            ("1%–2%", "1.341%", "0.318%", "98.5%"),
            ("高于 2%", "2.494%", "0.701%", "100%"),
        ],
        [2600, 2200, 2200, 2360],
    )
    add_callout(doc, "核心问题：预测过度收缩", "模型对小于 0.5% 的变化较合理，但几乎总是低估较大波动。下一阶段的主要任务不是继续增加相似线性模型，而是预测和校准波动幅度。", fill="FFF7E6", color=GOLD)
    for text in [
        "现有测试区间受到事后事件窗口定义影响。",
        "五个原始线性模型的预测高度相关，组合多样性不足。",
        "PCA 改善有限，并降低单个经济因子的解释性。",
        "入场时点尚需最终确认，决定能否使用开盘跳空和前 5–15 分钟信息。",
        "极端事件缺乏可在事前获得的高质量因子。",
    ]:
        add_bullet(doc, text)

    add_heading(doc, "19. 下一阶段计划", 1)
    for text in [
        "使用滚动样本外预测估计幅度校准系数。",
        "单独建立绝对波动幅度模型，标签为实际涨跌幅的绝对值。",
        "将连续收益预测与预测波动幅度组合。",
        "加入隔夜 XAU/USD、COMEX、USD/CNH、开盘跳空和开盘量价信息。",
        "针对不同波动率状态估计不同校准系数。",
        "保留新的未来数据作为完全独立的最终测试集。",
        "将预测误差连接至仓位、交易成本、持有期和 P&L 账本。",
    ]:
        add_number(doc, text)
    add_callout(doc, "最近的低风险改进", "对 PCA11 绝对误差模型进行滚动幅度校准。初步诊断表明约 1.42 倍的放大系数可能降低 RMSE，但必须在多个时间折中重新估计，不能直接固定使用。")

    add_heading(doc, "20. 十分组 Huber 交叉验证（已完成）", 1)
    add_para(doc, "全部 2,728 条可用建模观测已按日期顺序切分为十个连续数据块。每次保留一块测试，使用其余九块重新训练完整的中位数填补、标准化和 Huber Pipeline，最终保存十个独立模型，并让每条观测获得一次保留块预测。")
    for text in [
        "将观测划分为十个数据块。",
        "第 i 次将第 i 块作为测试块，其余九块作为训练块。",
        "每次重新拟合缺失值、中位数、标准化参数和 Huber 系数。",
        "保存十个模型和每个测试块的样本外预测。",
        "合并十个测试块，形成完整的样本外金额误差报告。",
    ]:
        add_number(doc, text)
    add_table(
        doc,
        ["结果口径", "十分组汇总"],
        [
            ("模型数量", "10"),
            ("每折测试观测", "272–273"),
            ("合并样本外观测", "2,728"),
            ("增长 MAE", "0.587 个百分点"),
            ("增长 RMSE", "0.928 个百分点"),
            ("波动幅度 MAE", "0.466 个百分点"),
            ("价格变化 MAE", "2.576 元/克"),
            ("折间增长 MAE 范围", "0.350–0.958 个百分点"),
        ],
        [3300, 6060],
    )
    add_callout(
        doc,
        "时间序列警告",
        "普通十折方法会让未来数据参与过去测试块的训练，产生未来信息泄漏。更安全的做法是十个滚动或扩展时间窗口：每个测试块只能使用其之前的历史数据训练。若必须严格执行九块训练一块测试，应将其明确标记为相关性诊断，而不是可交易的样本外回测。",
        fill="FDECEC",
        color=RED,
    )

    add_heading(doc, "21. Huber 损失函数数值优化与系数报告（已完成）", 1)
    add_para(doc, "为回应损失函数、偏导数和梯度下降的研究要求，项目使用全部 2,728 条事件过滤后观测，对当前 14 个特征的正则化 Huber 损失进行数值最小化。目标变量为从下一交易日开盘到预测期退出收盘的对数收益率。固定 epsilon=1.1、alpha=0.001，并使用带解析梯度的 L-BFGS-B 求解；该方法属于梯度型数值优化。")
    add_table(
        doc,
        ["排名", "特征", "标准化系数", "原始单位系数"],
        [
            ("1", "xauusd_return_1d", "0.002492", "0.284013"),
            ("2", "momentum_5d_x_volatility_20d", "-0.001084", "-3.863286"),
            ("3", "premium_change_5d", "-0.000925", "-0.094578"),
            ("4", "breakeven_inflation_10y", "0.000826", "0.002247"),
            ("5", "shanghai_gold_premium", "-0.000685", "-0.061942"),
            ("6", "momentum_5d", "0.000631", "0.032324"),
            ("7", "shfe_gold_settle_return_1d", "-0.000612", "-0.077971"),
            ("8", "return_1d", "0.000181", "0.021161"),
            ("9", "xauusd_momentum_5d", "0.000162", "0.008530"),
            ("10", "usdcnh_return_1d", "0.000161", "0.058439"),
            ("11", "volatility_20d", "0.000146", "0.031234"),
            ("12", "oil_momentum_5d", "-0.000123", "-0.002065"),
            ("13", "ma_gap_20d", "-0.000059", "-0.002813"),
            ("14", "momentum_20d", "-0.000037", "-0.001002"),
        ],
        [700, 4160, 2100, 2400],
    )
    add_table(
        doc,
        ["优化诊断", "结果"],
        [
            ("标准化截距", "-0.00074433"),
            ("估计残差尺度", "0.00160596"),
            ("最终 Huber 目标值", "33.7448368910"),
            ("零系数基准目标值", "37.3465299306"),
            ("相对零系数损失下降", "9.644%"),
            ("数值细化迭代", "16"),
            ("平均目标梯度无穷范数", "4.82e-08"),
            ("优化状态", "成功收敛"),
        ],
        [3600, 5760],
    )
    add_callout(
        doc,
        "解释限制",
        "标准化系数适合比较因子影响大小；原始单位系数只能结合各特征的单位解释。该模型使用全部样本估计系数，因此用于参数报告和部署，不得把它的拟合误差当作样本外测试表现。十折结果仍用于检验系数和预测误差的跨时期稳定性。",
        fill="FFF7E6",
        color=GOLD,
    )

    add_heading(doc, "22. 关键代码与输出", 1)
    add_table(
        doc,
        ["模块", "文件或目录"],
        [
            ("主模型", "gold_model_comparison.py"),
            ("固定期限比较", "compare_fixed_horizons.py"),
            ("交互项与状态", "interaction_regime_features.py；compare_interaction_regimes.py"),
            ("统计特征选择", "statistical_feature_selection.py"),
            ("Bagging", "compare_bagged_huber.py；compare_bagging_all_models.py"),
            ("PCA", "compare_pca_models.py；outputs/pca_h1/"),
            ("损失函数", "compare_loss_functions.py；outputs/loss_function_h1/"),
            ("加权平均", "compare_weighted_ensemble.py；outputs/weighted_ensemble_h1/"),
            ("金额误差报告", "build_growth_amount_report.py；outputs/growth_amount_report_h1/"),
            ("十分组 Huber", "huber_10fold_blocked_cv.py；outputs/huber_10fold_blocked/"),
            ("十九分组滚动 Huber", "huber_rolling_19block_cv.py；outputs/huber_rolling_19block/"),
            ("Huber 损失最小化", "huber_loss_coefficient_report.py；outputs/huber_loss_minimization/"),
            ("自动化测试", "tests/；.github/workflows/tests.yml"),
            ("QMT 可选集成", "qmt/"),
            ("项目笔记", "Gold_Model_Project_Notes.docx"),
        ],
        [2500, 6860],
    )

    add_heading(doc, "23. GitHub 版本管理与自动化测试（已完成）", 1)
    add_para(doc, "项目已连接至公开 GitHub 仓库 asuka20302/Gold-Research。现有研究成果作为基线版本纳入 Git；今后的模型、数据处理和回测修改应使用小而明确的提交记录。GitHub 不能恢复此前真实的开发历史，但可以从本次基线提交开始形成可审计的版本轨迹。")
    add_table(
        doc,
        ["目录", "内容与规则"],
        [
            ("项目根目录", "五模型主流程、统计筛选、PCA、损失函数、Bagging、加权及 P&L 代码"),
            ("docs/", "当前项目笔记与中文完整流程文档"),
            ("qmt/", "可选 QMT 信号导出和策略模板，与研究主流程隔离"),
            ("reports/", "供代码审查的小型最终结果表，不包含完整市场数据"),
            ("tests/", "目标时间、五模型范围、交互项、状态标签、Huber 梯度和十九分组公平性测试"),
            ("data/、outputs/", "本地下载和生成目录；默认不提交 Git"),
        ],
        [2400, 6960],
    )
    for text in [
        "删除已合并进文档的一次性更新脚本和临时渲染目录。",
        "使用 .gitignore 排除虚拟环境、密钥、下载数据、完整输出、模型文件和缓存。",
        "提供 .env.example、requirements.txt、README 和目录说明以便复现。",
        "新增 8 个 pytest 测试；本地测试全部通过。",
        "新增 GitHub Actions，在每次 push 和 pull request 时自动运行测试。",
    ]:
        add_bullet(doc, text)
    add_callout(
        doc,
        "版本管理规则",
        "不得提交真实 Tushare Token、账户信息、完整数据缓存或未经审查的大型模型输出。每个研究步骤使用独立、可解释的提交；测试失败时不得合并。",
        fill="FDECEC",
        color=RED,
    )

    add_heading(doc, "24. 十九分组固定窗口 Huber 验证（当前公平方案）", 1)
    add_para(doc, "为消除扩展窗口中训练样本数量不同带来的公平性问题，2,728 条可用观测被处理为 19 个完全相等的连续区块。每块 143 条，最早 11 条余数不进入该公平性诊断。模型 1 使用区块 1—9 训练、区块 10 测试；之后窗口每次向前移动一块，直到模型 10 使用区块 10—18 训练、区块 19 测试。")
    add_table(
        doc,
        ["设计或结果", "数值"],
        [
            ("模型数量", "10"),
            ("每个模型训练区块", "9"),
            ("每个模型训练观测", "1,287"),
            ("每个模型测试观测", "143"),
            ("合并严格样本外预测", "1,430"),
            ("增长 MAE", "0.639 个百分点"),
            ("增长 RMSE", "0.908 个百分点"),
            ("波动幅度 MAE", "0.517 个百分点"),
            ("价格变化 MAE", "3.531 元/克"),
        ],
        [3600, 5760],
    )
    for text in [
        "所有训练日期严格早于对应测试日期。",
        "每个训练目标在测试块首个信号日之前或当日已经可观察。",
        "十个模型固定使用相同 14 个特征、epsilon=1.1 和 alpha=0.001。",
        "测试窗口增长 MAE 范围为 0.428—0.817 个百分点。",
        "国际金价日收益、动量与波动率交互、溢价变化及通胀预期系数符号最稳定。",
    ]:
        add_bullet(doc, text)
    add_callout(
        doc,
        "与旧十分组结果的区别",
        "旧十分组方法的前九折会使用测试日期之后的数据，只适合相关性和系数稳定性诊断。十九分组固定窗口结果没有这种未来数据泄漏，且每个模型训练量相同，因此应作为当前主要验证结果。两个实验的测试日期和样本量不同，误差不能被解释为完全同口径的模型优劣。",
        fill="FFF7E6",
        color=GOLD,
    )

    add_heading(doc, "25. 当前结论", 1)
    add_para(doc, "项目已经完成从数据获取到金额误差账本的完整研究闭环。统计特征选择、稳健回归、PCA、绝对误差损失、移动区块 Bagging、样本外加权、十分组诊断以及十九分组固定滚动验证均已有可复现实现。")
    add_para(doc, "当前公平验证中的 Huber 增长 MAE 为 0.639 个百分点，但模型仍系统性低估较大波动，最近测试窗口的价格变化 MAE 明显较高。下一步应使用这些严格样本外预测确定交易阈值、持仓规模和交易成本后的 P&L，而不是继续引用存在未来信息的十分组误差。", bold=True, color=INK)

    # Keep table rows together where feasible and add document metadata.
    doc.core_properties.title = "黄金量化预测项目完整流程"
    doc.core_properties.subject = "Au99.99 量化预测项目研究流程手册"
    doc.core_properties.author = "Quant Research Project"
    doc.core_properties.keywords = "黄金, Au99.99, 回归, PCA, Huber, Bagging, 损失函数, 加权平均"

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()
