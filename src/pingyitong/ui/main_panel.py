"""主设置面板。

对应开发文档一·（五）"主设置面板"，以及产品功能规划四的用户交互流程：
"用户首次启动软件后，在主设置面板中选择目标语言与默认模型模式，
点击'开始翻译'按钮。"

面板结构
    顶部：运行状态条（状态点 + 模型 + 帧率 + 延迟 + 主操作按钮）
    中部：分页设置
        - 常规：语言、翻译层、启动行为
        - 模型：调度策略与模型状态
        - 样式：译文外观（含实时预览）
        - 术语库：导入导出与命中情况
        - 历史：记录策略
        - 关于：版本、开源组件、隐私声明
    底部：关闭提示（说明关闭后程序仍在后台运行）

设计说明
    - 所有设置**即时生效并持久化**，没有"保存/取消"按钮。
      桌面工具的设置项大多是"调一下看看效果"的性质，
      要求用户先点保存才能看到变化是多余的摩擦；
    - 破坏性操作（清空历史、恢复默认）需二次确认，且默认焦点在"取消"；
    - "关于"页完整列出开源组件与协议，对应原创性声明的要求。
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, replace

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..constants import (
    MODEL_ACCURATE,
    MODEL_FAST,
    MODEL_LABELS,
    SCHEDULE_MODE_AUTO,
    SCHEDULE_MODE_LABELS,
    SCHEDULE_MODE_MANUAL,
)
from ..glossary_catalog import BUILTIN_GLOSSARY_DOMAINS, DEFAULT_GLOSSARY_DOMAIN
from ..languages import test_summary
from ..utils.logger import get_logger
from ..version import version_string
from .design import LAYOUT, RADIUS, SPACE
from .design.theme import Theme, base_font
from .widgets import (
    Button,
    Card,
    ColorRow,
    FontRow,
    LanguageCombo,
    SegmentedControl,
    SettingRow,
    SliderRow,
    StatusPill,
    Switch,
)

logger = get_logger(__name__)

#: 在线 AI 服务商预设：``(显示名, 接口地址, 推荐模型名)``。
#:
#: 均为 OpenAI ``/chat/completions`` 兼容端点。模型名只作推荐
#: （各服务商的可用模型随时间变化），用户可在输入框中改成任何值；
#: 选"自定义"则三项全部留空，由用户手动填写。
API_PROVIDER_PRESETS: list[tuple[str, str, str]] = [
    ("自定义", "", ""),
    ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("智谱 GLM", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    ("月之暗面 Kimi", "https://api.moonshot.cn/v1", "kimi-k2-turbo-preview"),
    ("阿里通义千问", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini"),
    ("硅基流动", "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-7B-Instruct"),
    ("Ollama（本机）", "http://127.0.0.1:11434/v1", "qwen2.5:7b"),
]


class _ApiTestSignalHolder(QObject):
    """连通性测试的结果信号桥：后台线程 → 主线程。"""

    finished = pyqtSignal(object, bool, str)


class MainPanel(QWidget):
    """主设置面板。

    :param controller: 应用编排对象（:class:`~pingyitong.ui.app.Application`）。
        通过鸭子类型调用，避免与其产生循环导入。
    """

    def __init__(self, controller: object, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._theme: Theme = controller.theme  # type: ignore[attr-defined]
        self._settings = controller.settings  # type: ignore[attr-defined]

        self.setObjectName("Page")
        self.setWindowTitle("屏译通 · 设置")
        self.setMinimumSize(LAYOUT["panel_width"], LAYOUT["panel_height"])

        self._build_ui()
        self.load_from_settings()

    # ------------------------------------------------------------------
    # 构建界面
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """构建整体布局。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        # 分页
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(self._wrap_scroll(self._build_general_tab()), "常规")
        self._tabs.addTab(self._wrap_scroll(self._build_model_tab()), "模型")
        self._tabs.addTab(self._wrap_scroll(self._build_style_tab()), "样式")
        self._tabs.addTab(self._wrap_scroll(self._build_glossary_tab()), "术语库")
        self._tabs.addTab(self._wrap_scroll(self._build_history_tab()), "历史")
        self._tabs.addTab(self._wrap_scroll(self._build_about_tab()), "关于")
        root.addWidget(self._tabs, 1)

        root.addWidget(self._build_footer())

    def _wrap_scroll(self, content: QWidget) -> QScrollArea:
        """把分页内容放入滚动区域。"""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(content)
        return area

    def _build_header(self) -> QWidget:
        """顶部状态条。"""
        header = QWidget(self)
        header.setObjectName("Surface")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(SPACE["xl"], SPACE["lg"], SPACE["xl"], SPACE["lg"])
        layout.setSpacing(SPACE["md"])

        # 状态点 + 状态文字（颜色与文字并行，满足无障碍要求）
        self._status_pill = StatusPill("已暂停", "info")
        layout.addWidget(self._status_pill)

        self._model_pill = StatusPill("—", "info")
        layout.addWidget(self._model_pill)

        self._metrics_label = QLabel("—")
        self._metrics_label.setObjectName("Secondary")
        layout.addWidget(self._metrics_label)

        layout.addStretch(1)

        # 模型加载状态：常驻显示在开始按钮左侧。
        # 模型加载要数秒到十余秒，用户需要知道"现在能不能用"，
        # 而不是点了开始翻译却没反应。
        self._preload_label = QLabel("")
        self._preload_label.setObjectName("Secondary")
        layout.addWidget(self._preload_label)

        self._transcript_button = Button("文字栏", "secondary", self._theme)
        self._transcript_button.clicked.connect(self._on_toggle_transcript)
        self.set_transcript_open(False)
        layout.addWidget(self._transcript_button)

        self._toggle_button = Button("开始翻译", "primary", self._theme)
        self._toggle_button.clicked.connect(self._on_toggle_translation)
        layout.addWidget(self._toggle_button)

        return header

    def _build_footer(self) -> QWidget:
        """底部提示。"""
        footer = QWidget(self)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(SPACE["xl"], SPACE["md"], SPACE["xl"], SPACE["md"])

        hint = QLabel(
            "关闭本窗口后程序仍在后台运行，可通过系统托盘图标或全局热键操作。"
        )
        hint.setObjectName("Muted")
        layout.addWidget(hint)
        layout.addStretch(1)

        return footer

    # ------------------------------------------------------------------
    # 常规
    # ------------------------------------------------------------------

    def _build_general_tab(self) -> QWidget:
        """常规设置页。"""
        page = self._make_page()

        # 语言
        language_card = Card("语言", "自动识别适合中、英、日及繁体；其他语言请指定源语言。")
        self._source_language = self._make_language_combo(include_auto=True)
        language_card.add_row(
            SettingRow(
                "源语言",
                self._source_language,
                "无法识别时会说明原因；其他语言需手动指定。",
                self._theme,
            )
        )
        self._target_language = self._make_language_combo(include_auto=False)
        language_card.add_row(
            SettingRow("目标语言", self._target_language, "译文要翻译成的语言。", self._theme)
        )

        # 实测结论如实写在界面上：官方声称"支持 200+ 种语言"但未公布清单，
        # 本项目用固定探针句逐语言实测，把结果标在下拉项里并在此说明范围。
        self._language_test_label = QLabel(test_summary())
        self._language_test_label.setObjectName("Secondary")
        self._language_test_label.setWordWrap(True)
        language_card.add_row(
            SettingRow(
                "实测支持情况",
                self._language_test_label,
                "下拉框中的 ✓ 为固定句通过、~ 为疑似通过、✗ 为未通过；"
                "标记来自两个模型的固定句试译，不保证全部方向或 OCR。",
                self._theme,
            )
        )

        # 语义精简：只压缩措辞，不牺牲术语与信息
        self._concise_switch = Switch(False, self._theme)
        language_card.add_row(
            SettingRow(
                "原位精简",
                self._concise_switch,
                "原位使用经过审查的简短译文，文字栏和复制保留完整句子。"
                "空间不足时保留原文并用虚线提示，请在文字栏阅读完整译文。",
                self._theme,
            )
        )
        self._local_switch = Switch(True, self._theme)
        language_card.add_row(SettingRow(
            "仅本地处理", self._local_switch,
            "开启后屏幕文字只在本机处理，在线模型和联网预翻译暂停。", self._theme,
        ))
        self._review_switch = Switch(True, self._theme)
        language_card.add_row(SettingRow(
            "二次语义审查", self._review_switch,
            "重新理解原文并核对人物、动作、否定、条件、数字和遗漏；"
            "措辞有分歧时追加本地核对，无法确认时保留原文。首次翻译会增加等待。", self._theme,
        ))
        page.layout().addWidget(language_card)

        # 联网预翻译（方案 v1.2：开关只管抓取/预热，缓存直读常开）
        page.layout().addWidget(self._build_prefetch_card())

        # 翻译层
        overlay_card = Card("翻译层", "控制译文在屏幕上的呈现方式。")
        self._overlay_mode = SegmentedControl(
            ["完全覆盖", "半透明叠加"], current=0, theme=self._theme
        )
        overlay_card.add_row(
            SettingRow(
                "显示模式",
                self._overlay_mode,
                "完全覆盖会遮住原文；半透明叠加便于中英对照阅读。",
                self._theme,
            )
        )

        self._overlay_visible_switch = Switch(True, self._theme)
        overlay_card.add_row(
            SettingRow("显示翻译层", self._overlay_visible_switch, "", self._theme)
        )

        self._region_label = QLabel("全屏")
        self._region_label.setObjectName("Secondary")
        region_button = Button("框选区域", "secondary", self._theme)
        region_button.clicked.connect(lambda: self._controller.pick_region())
        region_row_control = self._make_inline(region_button, self._region_label)
        overlay_card.add_row(
            SettingRow(
                "翻译区域",
                region_row_control,
                "仅翻译屏幕的一部分可降低资源占用，并避免误翻译无关窗口。",
                self._theme,
            )
        )
        page.layout().addWidget(overlay_card)

        # 性能
        performance_card = Card("性能", "依据电脑配置调整，默认值适用于办公电脑。")
        self._fps_slider = SliderRow(
            "画面监视间隔", 50, 1000, self._settings.config.capture.monitor_interval_ms,
            self._theme, suffix=" ms", on_change=self._on_monitor_interval_changed,
        )
        performance_card.add_row(SettingRow("画面监视间隔", self._fps_slider,
            "默认每50ms查看变化；稳定画面不重复识别和推理。", self._theme, stack=True))

        self._diff_switch = Switch(self._settings.config.capture.enable_diff, self._theme)
        performance_card.add_row(
            SettingRow(
                "画面差分",
                self._diff_switch,
                "仅识别发生变化的屏幕区域。关闭后每帧都整屏识别，CPU 占用会显著上升。",
                self._theme,
            )
        )
        page.layout().addWidget(performance_card)

        # 系统集成
        system_card = Card("系统集成")
        self._start_minimized_switch = Switch(True, self._theme)
        system_card.add_row(
            SettingRow("启动后最小化到托盘", self._start_minimized_switch, "", self._theme)
        )

        self._notifications_switch = Switch(True, self._theme)
        system_card.add_row(
            SettingRow("显示通知", self._notifications_switch, "", self._theme)
        )

        # 减少动态效果：默认遵循系统设置
        self._reduced_motion_switch = Switch(False, self._theme)
        system_card.add_row(
            SettingRow(
                "减少动态效果",
                self._reduced_motion_switch,
                "界面过渡更简短、去掉位移与缩放动画。若系统已开启该设置则自动生效。",
                self._theme,
            )
        )
        page.layout().addWidget(system_card)

        # 热键状态：把注册结果直接展示出来。
        # 必须显示——热键被其他软件占用时"按了没反应"对用户是完全不可诊断的，
        # 而这是很常见的情况（截图工具、输入法、录屏软件都会抢这几个组合键）。
        hotkey_card = Card(
            "全局热键",
            "在任意应用中都可使用。若提示被占用，说明该组合键已被其他软件注册。",
        )

        self._fast_path_hotkeys = QLabel("检测中…")
        self._fast_path_hotkeys.setObjectName("Secondary")
        self._fast_path_hotkeys.setWordWrap(True)
        hotkey_card.add_row(
            SettingRow("注册状态", self._fast_path_hotkeys, "", self._theme, stack=True)
        )

        self._hotkey_hint = QLabel(
            "修改热键：编辑配置文件中的 hotkey 节后重启程序。"
        )
        self._hotkey_hint.setObjectName("Muted")
        self._hotkey_hint.setWordWrap(True)
        hotkey_card.add_widget(self._hotkey_hint)
        page.layout().addWidget(hotkey_card)

        page.layout().addStretch(1)
        self._connect_general_signals()
        return page

    def _build_prefetch_card(self) -> Card:
        """联网预翻译设置卡（方案 v1.2 §2/§3）。

        界面必须显示**实际生效模式**（状态机快照），不允许开关状态与
        实际行为不一致：徽标由控制器按网关快照驱动（`set_prefetch_badge`），
        而不是开关自己的勾选态。
        """
        prefetch = self._settings.config.prefetch
        card = Card(
            "联网预翻译",
            "在屏幕实时翻译之外，提前抓取并翻译字幕/网页内容，播放或阅读时"
            "可使用缓存。默认关闭；关闭或断网仍可读取本地模式的有效缓存，"
            "未命中的内容由屏幕实时翻译处理。",
        )

        self._prefetch_switch = Switch(prefetch.enabled, self._theme)
        card.add_row(
            SettingRow(
                "联网预翻译",
                self._prefetch_switch,
                "首次开启会询问授权；联网用于网络探测与获取指定静态网页。"
                "预翻译按当前模型配置执行，启用在线 AI 时待译文字会发送到配置的接口。",
                self._theme,
            )
        )

        self._prefetch_wifi_switch = Switch(prefetch.wifi_only, self._theme)
        card.add_row(
            SettingRow(
                "仅在 Wi-Fi 下预取",
                self._prefetch_wifi_switch,
                "识别失败的网络按计费处理（保守不预取）。"
                "注意：手机热点可能被 Windows 误报为不限费。",
                self._theme,
            )
        )

        # 生效徽标：实际行为，由控制器驱动
        self._prefetch_badge = StatusPill("已关闭", "info")
        card.add_row(SettingRow("当前生效模式", self._prefetch_badge, "", self._theme))

        # 预热进度：仅在有字幕轨预热时显示内容
        self._prefetch_progress = QLabel("")
        self._prefetch_progress.setObjectName("Secondary")
        card.add_widget(self._prefetch_progress)

        # 缓存占用 + 清除
        self._prefetch_usage = QLabel("缓存占用 0 KB")
        self._prefetch_usage.setObjectName("Secondary")
        clear_button = Button("清除缓存", "secondary", self._theme)
        clear_button.clicked.connect(self._on_clear_prefetch_cache)
        usage_row = self._make_inline(clear_button, self._prefetch_usage)
        card.add_row(
            SettingRow(
                "预翻译缓存",
                usage_row,
                "本地模式下有效缓存断网也可用；在线 AI 模式优先请求接口。"
                "容量超限自动按最久未用淘汰。",
                self._theme,
            )
        )

        self._prefetch_switch.toggled.connect(self._on_prefetch_toggled)
        self._prefetch_wifi_switch.toggled.connect(
            lambda value: self._update_config({"prefetch": {"wifi_only": value}})
        )
        return card

    def _on_prefetch_toggled(self, value: bool) -> None:
        """开关切换：首次开启先弹授权说明，拒绝则回落。"""
        config = self._settings.config.prefetch
        if value and not config.consent_granted:
            from PyQt6.QtWidgets import QMessageBox

            answer = QMessageBox.question(
                self,
                "开启联网预翻译",
                "开启后，软件会探测网络，并获取你指定的静态网页用于提前翻译：\n\n"
                "  · 定期探测网络可达性（固定探测点）\n"
                "  · 抓取你指定的静态网页正文（遵守 robots.txt、不带登录态、限速）\n"
                "  · 本地模型翻译不上传正文；启用在线 AI 后，待译文字会发送到配置的接口\n\n"
                "手机热点可能被 Windows 误判为不限费网络，"
                "热点场景请保持「仅在 Wi-Fi 下预取」开启。\n\n"
                "是否开启？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                blocked = self._prefetch_switch.blockSignals(True)
                self._prefetch_switch.setChecked(False)
                self._prefetch_switch.blockSignals(blocked)
                return
            self._update_config(
                {"prefetch": {"enabled": True, "consent_granted": True}}
            )
            return

        self._update_config({"prefetch": {"enabled": bool(value)}})

    def _on_clear_prefetch_cache(self) -> None:
        """清除预翻译缓存（控制器转发给应用装配的 SentenceCache）。"""
        self._controller.clear_prefetch_cache()
        self.set_prefetch_usage(self._controller.prefetch_cache_usage_text())

    def set_prefetch_badge(self, text: str) -> None:
        """更新生效模式徽标（由控制器按状态机快照驱动）。"""
        if hasattr(self, "_prefetch_badge"):
            self._prefetch_badge.setText(text)

    def set_prefetch_usage(self, text: str) -> None:
        """更新缓存占用展示。"""
        if hasattr(self, "_prefetch_usage"):
            self._prefetch_usage.setText(text)

    def set_prefetch_progress(self, text: str) -> None:
        """更新预热进度文案（空串隐藏）。"""
        if hasattr(self, "_prefetch_progress"):
            self._prefetch_progress.setText(text)
            self._prefetch_progress.setVisible(bool(text))

    def _connect_general_signals(self) -> None:
        """连接常规页信号。"""
        self._source_language.currentIndexChanged.connect(self._on_languages_changed)
        self._target_language.currentIndexChanged.connect(self._on_languages_changed)
        self._overlay_mode.selected_changed.connect(self._on_overlay_mode_changed)
        self._concise_switch.toggled.connect(
            lambda value: self._update_config({"translation": {"compact_display": value}})
        )
        self._local_switch.toggled.connect(
            lambda value: self._update_config({"translation": {"local_only": value}})
        )
        self._review_switch.toggled.connect(
            lambda value: self._update_config({"translation": {"semantic_review": value,
                                                                  "review_policy": "always"}})
        )
        self._overlay_visible_switch.toggled.connect(
            lambda value: self._controller.set_overlay_visible(value)
        )
        self._diff_switch.toggled.connect(
            lambda value: self._update_config({"capture": {"enable_diff": value}})
        )
        self._start_minimized_switch.toggled.connect(
            lambda value: self._update_config({"ui": {"start_minimized": value}})
        )
        self._notifications_switch.toggled.connect(self._on_notifications_changed)
        self._reduced_motion_switch.toggled.connect(self._on_reduced_motion_changed)

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------

    def _build_model_tab(self) -> QWidget:
        """模型设置页。"""
        page = self._make_page()
        schedule_config = self._settings.config.schedule

        schedule_card = Card(
            "调度策略",
            "双模型在速度与精度之间自适应平衡：1.7B 优先保证实时性，"
            "4B 提供更高的翻译质量。",
        )

        self._schedule_mode = SegmentedControl(
            [SCHEDULE_MODE_LABELS[SCHEDULE_MODE_MANUAL], SCHEDULE_MODE_LABELS[SCHEDULE_MODE_AUTO]],
            current=0 if schedule_config.mode == SCHEDULE_MODE_MANUAL else 1,
            theme=self._theme,
        )
        schedule_card.add_row(
            SettingRow(
                "调度模式",
                self._schedule_mode,
                "自动调度会依据 CPU 负载、GPU 显存与文本复杂度在两个模型间切换。",
                self._theme,
            )
        )

        self._model_switch = SegmentedControl(
            [MODEL_LABELS[MODEL_FAST], MODEL_LABELS[MODEL_ACCURATE]],
            current=0 if schedule_config.active_model == MODEL_FAST else 1,
            theme=self._theme,
        )
        self._manual_model_row = SettingRow(
            "手动指定模型",
            self._model_switch,
            "仅在调度模式为「手动」时生效。",
            self._theme,
        )
        schedule_card.add_row(self._manual_model_row)

        self._resource_adaptive_switch = Switch(schedule_config.resource_adaptive, self._theme)
        schedule_card.add_row(
            SettingRow(
                "资源自适应",
                self._resource_adaptive_switch,
                "CPU 持续高于阈值或渲染帧率过低时自动降级到 1.7B，优先保证桌面流畅。",
                self._theme,
            )
        )

        self._content_adaptive_switch = Switch(schedule_config.content_adaptive, self._theme)
        schedule_card.add_row(
            SettingRow(
                "内容自适应",
                self._content_adaptive_switch,
                "长句与专业术语密度高的文本自动使用 4B 模型。",
                self._theme,
            )
        )
        # 在线 AI 模型（可选）：置于模型页首位——它是接入外部大模型的
        # 入口，用户打开本页最常找的就是它；调度策略与模型状态依次排后。
        page.layout().addWidget(self._build_api_card())
        page.layout().addWidget(schedule_card)

        # 模型状态
        status_card = Card("模型状态", "将 GGUF 权重文件放入模型目录即可启用对应模型。")

        self._fast_status = StatusPill("检测中…", "info")
        status_card.add_row(
            SettingRow("Spark-X2.5-1.7B", self._fast_status, "极速模式，常驻内存。", self._theme)
        )

        self._accurate_status = StatusPill("检测中…", "info")
        status_card.add_row(
            SettingRow(
                "Spark-X2.5-4B",
                self._accurate_status,
                "高精度模式，按需加载、空闲自动卸载。",
                self._theme,
            )
        )

        open_models_button = Button("打开模型目录", "secondary", self._theme)
        open_models_button.clicked.connect(lambda: self._controller.open_models_dir())
        reload_button = Button("立即加载", "primary", self._theme)
        reload_button.clicked.connect(lambda: self._controller.refresh_models())
        status_card.add_widget(
            self._make_inline(open_models_button, reload_button, stretch_after=True)
        )
        page.layout().addWidget(status_card)

        # 模型文件位置：把实际解析结果展示出来，用户可据此自行排查
        # "为什么没加载到"——是文件位置不对，还是文件名不匹配。
        path_card = Card(
            "模型文件位置",
            "程序会依次在下列位置查找权重，并容忍文件名大小写差异。",
        )

        self._fast_path_label = self._make_path_label()
        path_card.add_row(
            SettingRow("Spark-X2.5-1.7B", self._fast_path_label, "", self._theme, stack=True)
        )

        self._accurate_path_label = self._make_path_label()
        path_card.add_row(
            SettingRow("Spark-X2.5-4B", self._accurate_path_label, "", self._theme, stack=True)
        )

        self._search_dir_label = self._make_path_label()
        path_card.add_row(
            SettingRow(
                "默认下载目录", self._search_dir_label,
                "运行 scripts/download_models.py 可自动下载到该目录；"
                "也可把 GGUF 文件手动放入该目录。",
                self._theme, stack=True,
            )
        )
        page.layout().addWidget(path_card)

        # 高级
        advanced_card = Card("高级", "改动会影响速度与内存占用，建议保持默认。")
        model_config = self._settings.config.model

        self._batch_slider = SliderRow(
            "批量翻译段数", 1, 16, model_config.batch_size, self._theme,
            on_change=lambda value: self._update_config(
                {"model": {"batch_size": int(value)}}
            ),
        )
        advanced_card.add_row(
            SettingRow(
                "批量翻译段数",
                self._batch_slider,
                "多个文本片段合并为一次推理。数值越大吞吐越高，但单次延迟也越大。",
                self._theme,
                stack=True,
            )
        )

        self._cache_switch = Switch(model_config.enable_cache, self._theme)
        advanced_card.add_row(
            SettingRow(
                "推理缓存",
                self._cache_switch,
                "相同文本直接命中缓存，界面静态文字反复出现时可显著提速。",
                self._theme,
            )
        )

        self._cache_stats_label = QLabel("—")
        self._cache_stats_label.setObjectName("Secondary")
        clear_cache_button = Button("清空缓存", "secondary", self._theme)
        clear_cache_button.clicked.connect(lambda: self._controller.clear_cache())
        advanced_card.add_row(
            SettingRow(
                "缓存状态",
                self._make_inline(clear_cache_button, self._cache_stats_label),
                "",
                self._theme,
            )
        )
        page.layout().addWidget(advanced_card)

        self._schedule_mode.selected_changed.connect(self._on_schedule_mode_changed)
        self._model_switch.selected_changed.connect(self._on_manual_model_changed)
        self._resource_adaptive_switch.toggled.connect(
            lambda value: self._update_config({"schedule": {"resource_adaptive": value}})
        )
        self._content_adaptive_switch.toggled.connect(
            lambda value: self._update_config({"schedule": {"content_adaptive": value}})
        )
        self._cache_switch.toggled.connect(
            lambda value: self._update_config({"model": {"enable_cache": value}})
        )

        page.layout().addStretch(1)
        return page

    # ------------------------------------------------------------------
    # 在线 AI 模型（OpenAI 兼容接口）
    # ------------------------------------------------------------------

    def _build_api_card(self) -> Card:
        """在线 AI 设置卡：填 Key 即接入任意 OpenAI 兼容服务。

        与本地模型的关系是"优先、非替代"：启用后翻译请求走云端，
        云端失败（断网、Key 失效、限流）自动回退本地模型，译文不断供。
        """
        api = self._settings.config.api
        card = Card(
            "在线 AI 模型（可选）",
            "填写任意 OpenAI 兼容服务的 API Key（DeepSeek、智谱、Kimi、通义、"
            "OpenAI 或本机 Ollama 等），翻译改由云端大模型承担：质量更高、"
            "上下文更长，多批并发时整帧更快。调用失败自动回退本地模型。"
            "API Key 明文保存在本机配置文件中。",
        )

        self._api_switch = Switch(api.enabled, self._theme)
        card.add_row(
            SettingRow(
                "启用在线 AI",
                self._api_switch,
                "启用后翻译优先走云端模型；关闭或配置不完整时使用本地模型。",
                self._theme,
            )
        )

        self._api_provider = QComboBox()
        for name, base_url, _model in API_PROVIDER_PRESETS:
            self._api_provider.addItem(name, userData=base_url)
        card.add_row(
            SettingRow(
                "服务商预设",
                self._api_provider,
                "选择后自动填入接口地址与推荐模型名，均可再手动修改。",
                self._theme,
            )
        )

        self._api_base_url = QLineEdit()
        self._api_base_url.setPlaceholderText("https://api.deepseek.com/v1")
        self._api_base_url.setClearButtonEnabled(True)
        card.add_row(
            SettingRow(
                "接口地址",
                self._api_base_url,
                "OpenAI 兼容的根地址，程序在其后拼接 /chat/completions。",
                self._theme,
            )
        )

        self._api_key = QLineEdit()
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setPlaceholderText("sk-…（本机 Ollama 可留空）")
        card.add_row(
            SettingRow(
                "API Key",
                self._api_key,
                "仅保存在本机配置文件，随翻译请求发送给你填写的服务商。",
                self._theme,
            )
        )

        self._api_model = QLineEdit()
        self._api_model.setPlaceholderText("deepseek-chat")
        self._api_model.setClearButtonEnabled(True)
        card.add_row(
            SettingRow(
                "模型名称",
                self._api_model,
                "服务商文档中列出的模型标识，例如 deepseek-chat、glm-4-flash。",
                self._theme,
            )
        )

        self._api_batch_slider = SliderRow(
            "单次请求片段数", 1, 24, api.batch_size, self._theme,
            on_change=lambda value: self._update_config(
                {"api": {"batch_size": int(value)}}
            ),
        )
        card.add_row(
            SettingRow(
                "在线批量段数",
                self._api_batch_slider,
                "云端模型上下文长，可一次携带更多片段，减少请求往返次数。",
                self._theme,
                stack=True,
            )
        )

        self._api_conc_slider = SliderRow(
            "并发请求数", 1, 8, api.max_concurrency, self._theme,
            on_change=lambda value: self._update_config(
                {"api": {"max_concurrency": int(value)}}
            ),
        )
        card.add_row(
            SettingRow(
                "并发请求数",
                self._api_conc_slider,
                "整帧多批时并发发送，整帧延迟约降为 1/N；过大易触发限流。",
                self._theme,
                stack=True,
            )
        )

        self._api_status = StatusPill("未启用", "info")
        card.add_row(SettingRow("当前状态", self._api_status, "", self._theme))

        self._api_test_button = Button("测试连接", "secondary", self._theme)
        self._api_test_button.clicked.connect(self._on_test_api)
        self._api_test_result = QLabel("")
        self._api_test_result.setObjectName("Secondary")
        self._api_test_result.setWordWrap(True)
        card.add_widget(
            self._make_inline(self._api_test_button, self._api_test_result, stretch_after=True)
        )

        # 连通性测试在后台线程执行，结果经信号回主线程更新控件
        self._api_test_holder = _ApiTestSignalHolder(self)
        self._api_test_holder.finished.connect(self._on_api_test_finished)

        self._api_switch.toggled.connect(self._on_api_switch_toggled)
        self._api_provider.currentIndexChanged.connect(self._on_api_provider_changed)
        self._api_base_url.editingFinished.connect(self._persist_api)
        self._api_key.editingFinished.connect(self._persist_api)
        self._api_model.editingFinished.connect(self._persist_api)
        return card

    def _on_api_switch_toggled(self, value: bool) -> None:
        self._persist_api()
        # 配置不完整时配置层会强制回落为关闭：把实际生效状态同步回界面
        effective = self._settings.config.api.enabled
        if effective != value:
            self._set_switch(self._api_switch, effective)
            self._controller.notify(
                "在线 AI 未启用",
                "请先填写接口地址与模型名称，再启用在线 AI",
                "warning",
                key="api-incomplete",
            )

    def _on_api_provider_changed(self, index: int) -> None:
        """选择预设：覆盖地址与模型名（Key 永远由用户手填）。"""
        if index < 0 or index >= len(API_PROVIDER_PRESETS):
            return
        _, base_url, model_name = API_PROVIDER_PRESETS[index]
        self._api_base_url.setText(base_url)
        self._api_model.setText(model_name)
        self._persist_api()

    def _persist_api(self) -> None:
        """把界面上的在线 AI 字段写入配置并刷新状态徽标。"""
        self._update_config(
            {
                "api": {
                    "enabled": self._api_switch.isChecked(),
                    "base_url": self._api_base_url.text().strip(),
                    "api_key": self._api_key.text().strip(),
                    "model_name": self._api_model.text().strip(),
                }
            }
        )
        self._refresh_api_status()

    def _refresh_api_status(self) -> None:
        # During construction the controller has not assigned its panel yet.
        # Later refreshes must preserve the live engine's failure/route state.
        if getattr(self._controller, "panel", None) is self:
            self._controller._refresh_api_runtime_status()
            return
        api = self._settings.config.api
        if not api.enabled:
            self._api_status.set_status("未启用", "info")
        elif not api.base_url or not api.model_name:
            self._api_status.set_status("配置不完整", "warning")
        else:
            self._api_status.set_status(f"已启用 · {api.model_name}", "success")

    def _on_test_api(self) -> None:
        """后台验证接口；仅本地模式不产生测试连接。"""
        if self._settings.config.translation.local_only:
            self._api_test_result.setText("仅本地处理已开启；关闭后才会连接外部接口")
            return
        self._persist_api()
        api = replace(self._settings.config.api)
        tested_settings = asdict(api)
        if not api.base_url or not api.model_name:
            self._api_test_result.setText("请先填写接口地址与模型名称")
            return
        holder = self._api_test_holder
        self._api_test_button.setEnabled(False)
        self._api_test_result.setText("测试中…")

        def _worker() -> None:
            # 已启用时复用翻译正在使用的引擎：连接测试成功即可解除其冷却。
            ok, message = self._controller.model_pool.test_api_connection(api)
            holder.finished.emit(tested_settings, ok, message)

        import threading

        threading.Thread(target=_worker, name="ApiTest", daemon=True).start()

    def _on_api_test_finished(self, tested_settings: object, ok: bool, message: str) -> None:
        self._api_test_button.setEnabled(True)
        if tested_settings != asdict(self._settings.config.api):
            self._api_test_result.setText("配置已变化，请重新测试")
            self._refresh_api_status()
            return
        self._api_test_result.setText(message)
        if self._settings.config.api.enabled:
            self._api_status.set_status(
                f"已启用 · {self._settings.config.api.model_name}" if ok else "连接失败 · 本地回退",
                "success" if ok else "warning",
            )
            if ok:
                self._controller.refresh_api_translation()

    def set_api_status(self, text: str, tone: str) -> None:
        """由控制器驱动在线 AI 状态徽标（如"冷却中，已回退本地"）。"""
        if hasattr(self, "_api_status"):
            self._api_status.set_status(text, tone)

    # ------------------------------------------------------------------
    # 样式
    # ------------------------------------------------------------------

    def _build_style_tab(self) -> QWidget:
        """译文样式页（含实时预览）。"""
        page = self._make_page()
        style = self._settings.config.style

        # 实时预览：所有改动立即反映在这里，用户不必切回桌面查看效果
        preview_card = Card("预览", "下方为译文在原文位置的实际呈现效果。")
        self._preview = _StylePreview(self._theme, self)
        preview_card.add_widget(self._preview)
        page.layout().addWidget(preview_card)

        # 字体
        font_card = Card("字体")
        self._font_row = FontRow(
            "字体", style.font_family, self._theme, on_change=self._on_font_changed
        )
        font_card.add_row(SettingRow("字体", self._font_row, "", self._theme, stack=True))

        self._font_size_mode = SegmentedControl(
            ["自动匹配原文", "固定字号"],
            current=0 if style.font_size_mode == "auto" else 1,
            theme=self._theme,
        )
        font_card.add_row(
            SettingRow(
                "字号模式",
                self._font_size_mode,
                "自动模式按原文包围盒高度推定字号，使译文与原文视觉大小一致。",
                self._theme,
            )
        )

        self._font_scale_slider = SliderRow(
            "字号缩放", 0.4, 3.0, style.font_scale, self._theme,
            suffix="×", decimals=1,
            on_change=lambda value: self._update_style({"font_scale": round(value, 2)}),
        )
        font_card.add_row(
            SettingRow("字号缩放", self._font_scale_slider, "", self._theme, stack=True)
        )

        self._bold_switch = Switch(style.bold, self._theme)
        self._bold_switch.toggled.connect(lambda value: self._update_style({"bold": value}))
        font_card.add_row(SettingRow("加粗", self._bold_switch, "", self._theme))
        page.layout().addWidget(font_card)

        # 颜色
        color_card = Card("颜色")
        self._text_color_row = ColorRow(
            "文字颜色", style.text_color, self._theme, on_change=self._on_text_color
        )
        color_card.add_row(
            SettingRow("文字颜色", self._text_color_row, "", self._theme, stack=True)
        )

        self._bg_color_row = ColorRow(
            "遮罩颜色", style.bg_color, self._theme, on_change=self._on_bg_color
        )
        color_card.add_row(
            SettingRow("遮罩颜色", self._bg_color_row, "", self._theme, stack=True)
        )

        self._bg_opacity_slider = SliderRow(
            "遮罩不透明度", 0.0, 1.0, style.bg_opacity, self._theme,
            suffix="%", scale=100.0,
            on_change=lambda value: self._update_style({"bg_opacity": round(value, 2)}),
        )
        color_card.add_row(
            SettingRow(
                "遮罩不透明度",
                self._bg_opacity_slider,
                "半透明叠加模式下使用；完全覆盖模式会自动使用不透明遮罩。",
                self._theme,
                stack=True,
            )
        )

        self._shadow_switch = Switch(style.shadow, self._theme)
        self._shadow_switch.toggled.connect(lambda value: self._update_style({"shadow": value}))
        color_card.add_row(
            SettingRow("文字阴影", self._shadow_switch, "深色文字配浅色背景时更容易读清。", self._theme)
        )
        page.layout().addWidget(color_card)

        # 排版
        layout_card = Card("排版")
        self._max_lines_slider = SliderRow(
            "最大行数", 1, 12, style.max_lines, self._theme,
            on_change=lambda value: self._update_style({"max_lines": int(value)}),
        )
        layout_card.add_row(
            SettingRow(
                "最大行数",
                self._max_lines_slider,
                "超出后截断并加省略号，避免译文溢出遮挡相邻内容。",
                self._theme,
                stack=True,
            )
        )

        self._radius_slider = SliderRow(
            "遮罩圆角", 0, 16, style.bg_radius, self._theme, suffix=" px",
            on_change=lambda value: self._update_style({"bg_radius": int(value)}),
        )
        layout_card.add_row(
            SettingRow("遮罩圆角", self._radius_slider, "", self._theme, stack=True)
        )
        page.layout().addWidget(layout_card)

        # 恢复默认
        reset_button = Button("恢复默认样式", "secondary", self._theme)
        reset_button.clicked.connect(self._on_reset_style)
        reset_row = QHBoxLayout()
        reset_row.addWidget(reset_button)
        reset_row.addStretch(1)
        reset_container = QWidget(self)
        reset_container.setLayout(reset_row)
        page.layout().addWidget(reset_container)

        page.layout().addStretch(1)
        return page

    # ------------------------------------------------------------------
    # 术语库
    # ------------------------------------------------------------------

    def _build_glossary_tab(self) -> QWidget:
        """术语库设置页。"""
        page = self._make_page()
        translation = self._settings.config.translation

        glossary_card = Card(
            "专业术语库",
            "按领域选择随软件安装的离线术语库，或导入客户自己的 CSV；"
            "术语仅在匹配的语言方向启用。",
        )

        self._glossary_switch = Switch(translation.glossary_enabled, self._theme)
        self._glossary_switch.toggled.connect(self._on_glossary_toggled)
        glossary_card.add_row(SettingRow("启用术语库", self._glossary_switch, "", self._theme))

        mode = translation.glossary_mode or (
            "custom" if translation.glossary_path else "builtin"
        )
        self._glossary_mode = SegmentedControl(
            ["预装词库", "自定义 CSV"],
            current=1 if mode == "custom" else 0,
            theme=self._theme,
        )
        self._glossary_mode.selected_changed.connect(self._on_glossary_mode_changed)
        glossary_card.add_row(SettingRow(
            "当前术语库", self._glossary_mode,
            "预装库随软件安装；可导入自己的 CSV 并随时切回。", self._theme,
        ))

        self._glossary_domain = QComboBox()
        for domain, label in BUILTIN_GLOSSARY_DOMAINS.items():
            self._glossary_domain.addItem(label, domain)
        self._glossary_domain.currentIndexChanged.connect(self._on_glossary_domain_changed)
        self._select_combo(
            self._glossary_domain,
            translation.glossary_domain or DEFAULT_GLOSSARY_DOMAIN,
        )
        self._glossary_domain.setEnabled(mode != "custom")
        glossary_card.add_row(SettingRow(
            "预装领域", self._glossary_domain,
            "医疗等专业词仅在所选领域启用，避免跨学科同形词误替换。", self._theme,
        ))

        self._glossary_stats = QLabel("未加载术语")
        self._glossary_stats.setObjectName("Secondary")
        glossary_card.add_row(
            SettingRow("当前状态", self._glossary_stats, "", self._theme)
        )

        import_button = Button("导入 CSV…", "primary", self._theme)
        import_button.clicked.connect(lambda: self._controller.import_glossary())
        export_button = Button("导出当前术语", "secondary", self._theme)
        export_button.clicked.connect(lambda: self._controller.export_glossary())
        sample_button = Button("打开示例文件", "ghost", self._theme)
        sample_button.clicked.connect(lambda: self._controller.open_sample_glossary())

        glossary_card.add_widget(self._make_inline(import_button, export_button, sample_button, stretch_after=True))
        page.layout().addWidget(glossary_card)

        hint_card = Card("CSV 格式说明")
        hint = QLabel(
            "文件需为 UTF-8 或 GBK 编码，每行一条术语，两列分别为原文与译文：\n\n"
            "    原文,译文\n"
            "    machine learning,机器学习\n"
            "    Spark-X2.5,讯飞星火 X2.5\n\n"
            "可选第三列指定是否区分大小写，第四列指定是否整词匹配（填「是」或「否」）。\n"
            "第五、六列可指定源语言和目标语言代码；留空则对所有语言方向生效。\n"
            "术语按长度从长到短匹配，因此「machine learning」不会被「machine」提前拆开。"
        )
        hint.setObjectName("Secondary")
        hint.setWordWrap(True)
        hint_card.add_widget(hint)
        page.layout().addWidget(hint_card)

        page.layout().addStretch(1)
        return page

    # ------------------------------------------------------------------
    # 历史
    # ------------------------------------------------------------------

    def _build_history_tab(self) -> QWidget:
        """翻译历史设置页。"""
        page = self._make_page()
        history = self._settings.config.history

        history_card = Card(
            "翻译历史",
            "每次翻译的原文与译文会按时间保存到本机数据库中，可检索与导出。",
        )

        self._history_switch = Switch(history.enabled, self._theme)
        self._history_switch.toggled.connect(
            lambda value: self._update_config({"history": {"enabled": value}})
        )
        history_card.add_row(SettingRow("记录翻译历史", self._history_switch, "", self._theme))

        self._store_source_switch = Switch(history.store_source_text, self._theme)
        self._store_source_switch.toggled.connect(
            lambda value: self._update_config({"history": {"store_source_text": value}})
        )
        history_card.add_row(
            SettingRow(
                "同时记录原文",
                self._store_source_switch,
                "原文来自屏幕，可能比译文包含更多敏感信息（如内部系统路径、账号名）。"
                "关闭后只保留译文。",
                self._theme,
            )
        )

        self._history_limit_slider = SliderRow(
            "最多保留条数", 1000, 100000, history.max_entries, self._theme,
            on_change=lambda value: self._update_config(
                {"history": {"max_entries": int(value)}}
            ),
        )
        history_card.add_row(
            SettingRow(
                "最多保留条数",
                self._history_limit_slider,
                "超出后自动删除最早的记录。",
                self._theme,
                stack=True,
            )
        )

        self._history_size_label = QLabel("—")
        self._history_size_label.setObjectName("Secondary")
        open_history_button = Button("打开历史记录…", "secondary", self._theme)
        open_history_button.clicked.connect(lambda: self._controller.open_history())
        history_card.add_row(
            SettingRow(
                "数据库",
                self._make_inline(open_history_button, self._history_size_label),
                "",
                self._theme,
            )
        )
        page.layout().addWidget(history_card)

        privacy_card = Card("隐私说明")
        privacy = QLabel(
            "历史数据库位于本机应用数据目录，不会自动上传或同步。"
            "启用在线 AI 时，当前待译文字会发送到配置的接口；程序无遥测上报。"
        )
        privacy.setObjectName("Secondary")
        privacy.setWordWrap(True)
        privacy_card.add_widget(privacy)
        page.layout().addWidget(privacy_card)

        page.layout().addStretch(1)
        return page

    # ------------------------------------------------------------------
    # 关于
    # ------------------------------------------------------------------

    def _build_about_tab(self) -> QWidget:
        """关于页：版本、开源组件、隐私声明。"""
        page = self._make_page()

        about_card = Card()
        title = QLabel("屏译通")
        title.setObjectName("Title")
        about_card.add_widget(title)

        subtitle = QLabel("支持本地与在线 AI 的桌面实时原位翻译系统")
        subtitle.setObjectName("Secondary")
        about_card.add_widget(subtitle)

        version_label = QLabel(f"版本 {version_string()}")
        version_label.setObjectName("Muted")
        about_card.add_widget(version_label)
        page.layout().addWidget(about_card)

        privacy_card = Card(
            "本地处理与可选联网",
            "屏幕捕获、文字识别与原位渲染在本机完成，本地模型翻译无需联网。"
            "启用在线 AI 后，待译文字会发送到配置的接口；启用联网预翻译后，"
            "会探测网络并获取指定静态网页。程序无遥测上报。",
        )
        page.layout().addWidget(privacy_card)

        components_card = Card(
            "开源组件",
            "本项目优先选用 GitHub 开源社区组件，并严格遵循各自的开源许可协议。",
        )
        for name, purpose, license_text in _OPEN_SOURCE_COMPONENTS:
            row = QLabel(f"<b>{name}</b> · {purpose} · {license_text}")
            row.setObjectName("Secondary")
            row.setWordWrap(True)
            components_card.add_widget(row)
        page.layout().addWidget(components_card)

        design_card = Card(
            "界面设计来源",
            "界面设计与动效规范参考 mozilla/OpenDesign（开放设计原则，"
            "设计作品以 CC0-1.0 释出）与 emilkowalski/skills（设计工程规范）。"
            "本项目的设计令牌与规范文档一并随源码交付，可自由审阅修改。",
        )
        page.layout().addWidget(design_card)

        page.layout().addStretch(1)
        return page

    # ------------------------------------------------------------------
    # 通用辅助
    # ------------------------------------------------------------------

    def _make_page(self) -> QWidget:
        """创建带统一边距与间距的分页容器。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        layout.setSpacing(SPACE["lg"])
        return page

    def _make_path_label(self) -> QLabel:
        """创建用于展示文件路径的标签。

        路径需要可选中复制——用户往往要把它粘到文件管理器里，
        不可选中的路径文本等于没给。
        """
        label = QLabel("检测中…")
        label.setObjectName("Muted")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _make_inline(self, *widgets: QWidget, stretch_after: bool = False) -> QWidget:
        """把多个控件横向排布为一个组合控件。"""
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["sm"])

        if stretch_after:
            layout.addStretch(0)

        for widget in widgets:
            layout.addWidget(widget)

        if stretch_after:
            layout.addStretch(1)
        return container

    def _make_language_combo(self, include_auto: bool) -> LanguageCombo:
        """创建语言下拉框。

        使用 :class:`LanguageCombo` 而不是原生 ``QComboBox``：
        可选语言有一百多种，需要**可搜索**，并且每一项要标出实测得到的
        支持档位（确定 / 疑似 / 不支持），用户不必逐个试错。
        """
        combo = LanguageCombo(
            current="auto" if include_auto else self._settings.config.translation.target_language,
            theme=self._theme,
            with_auto=include_auto,
        )
        # 少于一屏的高度更易用；下拉列表本身支持滚动
        combo.setMinimumWidth(220)
        return combo

    # ------------------------------------------------------------------
    # 配置读写
    # ------------------------------------------------------------------

    def load_from_settings(self) -> None:
        """把配置值同步到界面控件。

        全部 ``setChecked`` 都必须屏蔽信号：本方法在信号连接**之后**调用
        （界面构建完成时、以及恢复默认样式时），开关的构造值是硬编码默认值，
        一旦用户配置与其不同，``setChecked`` 会触发 ``toggled``，把刚读进来
        的配置原样写回并弹出伪通知（例如每次启动都弹"翻译层已隐藏"）。
        """
        config = self._settings.config

        self._select_combo(self._source_language, config.translation.source_language)
        self._select_combo(self._target_language, config.translation.target_language)

        self._overlay_mode.set_current_index(
            0 if config.style.mode == "cover" else 1, animate=False
        )
        self._set_switch(self._overlay_visible_switch, config.ui.overlay_visible)
        self._set_switch(self._concise_switch, config.translation.compact_display)
        self._set_switch(self._local_switch, config.translation.local_only)
        self._set_switch(self._review_switch, config.translation.semantic_review)
        self._set_switch(self._prefetch_switch, config.prefetch.enabled)
        self._set_switch(self._prefetch_wifi_switch, config.prefetch.wifi_only)
        self._set_switch(self._notifications_switch, config.ui.notifications)
        self._set_switch(self._reduced_motion_switch, config.ui.reduced_motion)

        self._fps_slider.set_value(config.capture.monitor_interval_ms)
        self._set_switch(self._diff_switch, config.capture.enable_diff)
        self._set_switch(self._start_minimized_switch, config.ui.start_minimized)

        self._font_row.set_font_family(config.style.font_family)
        self._font_size_mode.set_current_index(
            0 if config.style.font_size_mode == "auto" else 1, animate=False
        )
        self._font_scale_slider.set_value(config.style.font_scale)
        self._set_switch(self._bold_switch, config.style.bold)
        self._text_color_row.set_color(config.style.text_color, notify=False)
        self._bg_color_row.set_color(config.style.bg_color, notify=False)
        self._bg_opacity_slider.set_value(config.style.bg_opacity)
        self._set_switch(self._shadow_switch, config.style.shadow)
        self._max_lines_slider.set_value(config.style.max_lines)
        self._radius_slider.set_value(config.style.bg_radius)

        self._set_switch(self._glossary_switch, config.translation.glossary_enabled)
        self.set_glossary_mode(
            config.translation.glossary_mode or (
                "custom" if config.translation.glossary_path else "builtin"
            )
        )
        self._select_combo(
            self._glossary_domain,
            config.translation.glossary_domain or DEFAULT_GLOSSARY_DOMAIN,
        )
        self._set_switch(self._history_switch, config.history.enabled)
        self._set_switch(self._store_source_switch, config.history.store_source_text)
        self._history_limit_slider.set_value(config.history.max_entries)

        # 在线 AI：文本框与预设下拉都要屏蔽信号，避免加载即写回
        api = config.api
        self._set_switch(self._api_switch, api.enabled)
        for line_edit, value in (
            (self._api_base_url, api.base_url),
            (self._api_key, api.api_key),
            (self._api_model, api.model_name),
        ):
            blocked = line_edit.blockSignals(True)
            line_edit.setText(value)
            line_edit.blockSignals(blocked)
        index = self._api_provider.findData(api.base_url)
        blocked = self._api_provider.blockSignals(True)
        self._api_provider.setCurrentIndex(index if index >= 0 else 0)
        self._api_provider.blockSignals(blocked)
        self._api_batch_slider.set_value(api.batch_size)
        self._api_conc_slider.set_value(api.max_concurrency)
        self._refresh_api_status()

        self._preview.set_style(config.style)

    @staticmethod
    def _set_switch(switch, value: bool) -> None:
        """设置开关状态且不触发 ``toggled``（见 :meth:`load_from_settings`）。"""
        blocked = switch.blockSignals(True)
        switch.setChecked(bool(value))
        switch.blockSignals(blocked)

    def _select_combo(self, combo: QComboBox, value: str) -> None:
        """按数据值选中下拉项。"""
        index = combo.findData(value)
        if index >= 0:
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(blocked)

    def _update_config(self, patch: dict) -> None:
        """写入配置。

        ``settings.update`` 内部会广播 ``changed`` 信号，控制器连接了该信号
        并完成全部配置应用；这里若再显式调用一次 ``on_settings_changed``，
        同一个补丁会被完整应用两遍（拖动滑块时每一步都翻倍）。
        """
        self._settings.update(patch)

    def _update_style(self, patch: dict) -> None:
        """只更新样式并刷新预览。"""
        self._settings.update({"style": patch})
        self._preview.set_style(self._settings.config.style)

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def _on_toggle_translation(self) -> None:
        """主操作按钮：开始 / 暂停翻译。"""
        self._controller.toggle_translation()

    def _on_toggle_transcript(self) -> None:
        """以实时文字栏阅读屏幕译文。"""
        self._controller.toggle_transcript_panel()

    def _on_languages_changed(self) -> None:
        """语言选择变更。"""
        self._update_config(
            {
                "translation": {
                    "source_language": self._source_language.currentData(),
                    "target_language": self._target_language.currentData(),
                }
            }
        )
        # Show the effective configuration after saving the user's selections.
        translation = self._settings.config.translation
        self._select_combo(self._source_language, translation.source_language)
        self._select_combo(self._target_language, translation.target_language)

    def _on_overlay_mode_changed(self, index: int) -> None:
        """翻译层显示模式变更。"""
        self._update_style({"mode": "cover" if index == 0 else "overlay"})

    def _on_fps_changed(self, value: float) -> None:
        """帧率变更。"""
        self._update_config({"capture": {"fps": int(value)}})

    def _on_monitor_interval_changed(self, value: float) -> None:
        self._update_config({"capture": {"monitor_interval_ms": int(value)}})

    def _on_notifications_changed(self, value: bool) -> None:
        """通知开关变更。"""
        self._update_config({"ui": {"notifications": value}})
        self._controller.set_notifications(value)

    def _on_reduced_motion_changed(self, value: bool) -> None:
        """减少动态效果开关变更。"""
        self._update_config({"ui": {"reduced_motion": value}})
        self._controller.set_reduced_motion(value)

    def _on_schedule_mode_changed(self, index: int) -> None:
        """调度模式变更。"""
        mode = SCHEDULE_MODE_MANUAL if index == 0 else SCHEDULE_MODE_AUTO
        self._update_config({"schedule": {"mode": mode}})
        # 自动调度下手动选择无意义，因此禁用该行
        self._manual_model_row.setEnabled(mode == SCHEDULE_MODE_MANUAL)

    def _on_manual_model_changed(self, index: int) -> None:
        """手动指定模型。

        持久化统一由 set_model 负责：它内部会写 settings 并触发一次
        全量应用。这里再发同一补丁会造成同配置两次落盘、两轮全量应用
        （UI 审查 F6）。
        """
        model_id = MODEL_FAST if index == 0 else MODEL_ACCURATE
        self._controller.set_model(model_id)

    def _on_font_changed(self, family: str) -> None:
        """字体变更。"""
        self._update_style({"font_family": family})

    def _on_text_color(self, color: str) -> None:
        """文字颜色变更。"""
        self._update_style({"text_color": color})

    def _on_bg_color(self, color: str) -> None:
        """遮罩颜色变更。"""
        self._update_style({"bg_color": color})

    def _on_glossary_toggled(self, value: bool) -> None:
        """术语库开关变更。"""
        self._update_config({"translation": {"glossary_enabled": value}})
        self._controller.reload_glossary()

    def _on_glossary_mode_changed(self, index: int) -> None:
        mode = "custom" if index == 1 else "builtin"
        if not self._controller.select_glossary_mode(mode):
            translation = self._settings.config.translation
            self.set_glossary_mode(
                translation.glossary_mode or (
                    "custom" if translation.glossary_path else "builtin"
                )
            )

    def _on_glossary_domain_changed(self, index: int) -> None:
        domain = self._glossary_domain.itemData(index)
        if not self._controller.select_glossary_domain(domain):
            self._select_combo(
                self._glossary_domain,
                self._settings.config.translation.glossary_domain or DEFAULT_GLOSSARY_DOMAIN,
            )

    def set_glossary_mode(self, mode: str) -> None:
        blocked = self._glossary_mode.blockSignals(True)
        self._glossary_mode.set_current_index(1 if mode == "custom" else 0, animate=False)
        self._glossary_mode.blockSignals(blocked)
        self._glossary_domain.setEnabled(mode != "custom")

    def _on_reset_style(self) -> None:
        """恢复默认样式。

        ``reset_section`` 已广播变更并由控制器应用；``load_from_settings``
        自身屏蔽信号，不会造成回写。
        """
        self._settings.reset_section("style")
        self.load_from_settings()
        self._controller.notify("样式已恢复默认", "", "success")

    # ------------------------------------------------------------------
    # 外部状态同步
    # ------------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        """同步运行状态。"""
        self._toggle_button.setText("暂停翻译" if running else "开始翻译")
        self._status_pill.set_status("翻译中" if running else "已暂停", "success" if running else "info")

    def set_transcript_open(self, expanded: bool) -> None:
        """同步所有入口和窗口关闭引起的文字栏状态变化。"""
        self._transcript_button.setText("收起文字栏" if expanded else "文字栏")
        self._transcript_button.setAccessibleName("收起文字栏" if expanded else "展开文字栏")
        self._transcript_button.setToolTip(
            "收起文字栏并恢复原位译文" if expanded else "以文字栏阅读屏幕译文"
        )

    def set_model(self, model_id: str) -> None:
        """同步模型显示。

        本方法在自动调度切换模型时由状态刷新调用。必须屏蔽信号：
        分段控件的 ``selected_changed`` 连着"手动指定模型"处理器，
        不屏蔽的话程序选定的模型会被当成用户手选写回配置，
        每次自动切换都触发两轮配置应用与一次写盘。
        """
        self._model_pill.set_status(MODEL_LABELS.get(model_id, model_id), "info")
        blocked = self._model_switch.blockSignals(True)
        self._model_switch.set_current_index(
            0 if model_id == MODEL_FAST else 1, animate=False
        )
        self._model_switch.blockSignals(blocked)

    def set_metrics(self, text: str) -> None:
        """同步性能指标文本。"""
        self._metrics_label.setText(text)

    def set_preload_status(self, text: str, tone: str = "info") -> None:
        """同步模型加载状态。

        :param text: 状态文本；空字符串表示隐藏。
        :param tone: ``info``（加载中）/ ``success``（已就绪）/ ``warning``（未就绪）。
        """
        self._preload_label.setText(text)
        object_names = {
            "info": "Secondary",
            "success": "BadgeSuccess",
            "warning": "BadgeWarning",
        }
        self._preload_label.setObjectName(object_names.get(tone, "Secondary"))

        style = self._preload_label.style()
        if style is not None:
            style.unpolish(self._preload_label)
            style.polish(self._preload_label)
        self._preload_label.update()

    def set_region_text(self, text: str) -> None:
        """同步翻译区域描述。"""
        self._region_label.setText(text)

    def set_model_status(self, model_id: str, text: str, tone: str) -> None:
        """同步模型可用状态。"""
        pill = self._fast_status if model_id == MODEL_FAST else self._accurate_status
        pill.set_status(text, tone)

    def set_model_paths(self, resolved: dict[str, str], search_dir: str) -> None:
        """同步模型文件的解析路径。

        :param resolved: ``{模型标识: 解析到的路径或"未找到"}``。
        :param search_dir: 默认下载目录，供用户放置权重时参考。
        """
        for model_id, label in (
            (MODEL_FAST, self._fast_path_label),
            (MODEL_ACCURATE, self._accurate_path_label),
        ):
            path = resolved.get(model_id, "未找到")
            if path == "未找到":
                label.setText("未找到 —— 请下载权重或检查文件名")
                label.setObjectName("Danger")
            else:
                label.setText(path)
                label.setObjectName("Secondary")
            # 更换 objectName 后需重新应用样式表
            style = label.style()
            if style is not None:
                style.unpolish(label)
                style.polish(label)
            label.update()

        self._search_dir_label.setText(search_dir)

    def set_cache_stats(self, text: str) -> None:
        """同步缓存统计。"""
        self._cache_stats_label.setText(text)

    def set_glossary_stats(self, text: str) -> None:
        """同步术语库统计。"""
        self._glossary_stats.setText(text)

    def set_history_size(self, text: str) -> None:
        """同步历史数据库大小。"""
        self._history_size_label.setText(text)

    def set_overlay_visible_switch(self, visible: bool) -> None:
        """同步翻译层开关的勾选状态（不触发回调）。"""
        blocked = self._overlay_visible_switch.blockSignals(True)
        self._overlay_visible_switch.setChecked(visible)
        self._overlay_visible_switch.blockSignals(blocked)

    def set_hotkey_status(self, bindings: list, all_failed: bool = False) -> None:
        """显示热键注册结果。

        :param bindings: :class:`~pingyitong.services.hotkey_service.HotkeyBinding`
            列表；为空表示热键功能未启用或平台不支持。
        :param all_failed: 已尝试注册但一个都没成功。此时 ``bindings``
            必然为空，若沿用"未启用"文案，用户会以为功能正常、只是没配，
            完全意识不到热键正被其他程序占用。
        """
        from ..services.hotkey_service import ACTION_LABELS, format_sequence

        if all_failed:
            self._fast_path_hotkeys.setText("热键全部注册失败（可能被其他程序占用）")
            self._fast_path_hotkeys.setObjectName("Danger")
            self._hotkey_hint.setText(
                "所有全局热键都未能注册，按热键不会有任何反应。\n"
                "常见原因是截图工具、输入法、录屏软件占用了 Ctrl+Alt+字母 组合；"
                "可编辑配置文件（设置 → 关于 里可查看路径）的 hotkey 节改用其他"
                "组合，例如 Ctrl+Alt+F1，改完重启程序生效。"
            )
            self._hotkey_hint.setObjectName("Danger")
        elif not bindings:
            self._fast_path_hotkeys.setText("未启用（或当前平台不支持全局热键）")
            self._fast_path_hotkeys.setObjectName("Muted")
        else:
            lines: list[str] = []
            conflicts: list[str] = []
            for binding in bindings:
                label = ACTION_LABELS.get(binding.action, binding.action)
                sequence = format_sequence(binding.sequence)
                if binding.registered:
                    suffix = (
                        f"（原 {format_sequence(binding.fallback_from)} 被占用，已自动改用）"
                        if binding.fallback_from else ""
                    )
                    lines.append(f"✓ {sequence} — {label}{suffix}")
                else:
                    lines.append(f"✗ {sequence} — {label}（{binding.error or '注册失败'}）")
                    conflicts.append(sequence)

            self._fast_path_hotkeys.setText("\n".join(lines))
            self._fast_path_hotkeys.setObjectName("Danger" if conflicts else "Secondary")

            if conflicts:
                self._hotkey_hint.setText(
                    f"以下热键被其他软件占用，程序无法响应：{'、'.join(conflicts)}。\n"
                    "可编辑配置文件（设置 → 关于 里可查看路径，或用「打开模型目录」"
                    "同级目录）的 hotkey 节改用其他组合，例如 Ctrl+Alt+F1，"
                    "改完重启程序生效。"
                )
                self._hotkey_hint.setObjectName("Danger")
            else:
                self._hotkey_hint.setText(
                    "热键已自动切换到可用组合，配置文件未改动。"
                    if any(binding.fallback_from for binding in bindings)
                    else "全部热键已成功注册。"
                )
                self._hotkey_hint.setObjectName("Muted")

        for label in (self._fast_path_hotkeys, self._hotkey_hint):
            style = label.style()
            if style is not None:
                style.unpolish(label)
                style.polish(label)
            label.update()

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        for widget in self.findChildren(QWidget):
            if hasattr(widget, "set_theme"):
                with contextlib.suppress(Exception):
                    # 单个控件失败不应阻断整体换肤
                    widget.set_theme(theme)  # type: ignore[attr-defined]
        self._preview.set_theme(theme)
        self._preview.update()


# ----------------------------------------------------------------------
# 样式预览
# ----------------------------------------------------------------------


class _StylePreview(QWidget):
    """译文样式实时预览。

    模拟一块"桌面内容"，在原文位置绘制译文，让用户直接看到样式效果，
    不必切回桌面观察。预览使用与正式渲染层相同的绘制逻辑参数
    （配色、圆角、内边距、字号比例），因此所见即所得。
    """

    #: 模拟的原文文案
    SAMPLE_LINES = (
        "Documentation",
        "The quick brown fox jumps over the lazy dog.",
        "配置文件与术语表均保存在本机。",
    )
    #: 对应的模拟译文
    SAMPLE_TRANSLATIONS = (
        "文档",
        "敏捷的棕色狐狸跳过了懒狗。",
        "Configuration files and glossaries are stored locally.",
    )

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._style = None
        self.setFixedHeight(132)

    def set_style(self, style: object) -> None:
        """设置要预览的样式。"""
        self._style = style
        self.update()

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """绘制模拟桌面与译文块。"""
        if self._style is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        try:
            scheme = self._theme.scheme

            # 模拟的"桌面背景"
            canvas = self.rect().adjusted(0, 0, -1, -1)
            painter.setPen(QColor(scheme.border))
            painter.setBrush(QColor(scheme.background))
            painter.drawRoundedRect(canvas, RADIUS["sm"], RADIUS["sm"])

            padding = SPACE["md"]
            row_height = (canvas.height() - padding * 2) // len(self.SAMPLE_TRANSLATIONS)

            for index, translation in enumerate(self.SAMPLE_TRANSLATIONS):
                block = canvas.adjusted(padding, padding + index * row_height,
                                        -padding, -(canvas.height() - padding - (index + 1) * row_height))

                self._paint_block(painter, block, translation, scheme)

            # 预览标识
            painter.setFont(base_font(self._theme, "caption"))
            painter.setPen(QColor(scheme.text_muted))
            painter.drawText(
                canvas.adjusted(0, 0, -padding, -SPACE["xs"]),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom),
                "预览",
            )
        finally:
            painter.end()

    def _paint_block(self, painter: QPainter, rect, text: str, scheme) -> None:
        """绘制单个译文块，参数与渲染层保持一致。"""
        style = self._style
        radius = float(getattr(style, "bg_radius", 4))
        padding_x = float(getattr(style, "padding_x", 3))
        padding_y = float(getattr(style, "padding_y", 1))

        mode = getattr(style, "mode", "cover")
        base = QColor(getattr(style, "bg_color", "#101418"))
        if not base.isValid():
            base = QColor("#101418")

        opacity = float(getattr(style, "bg_opacity", 0.62))
        base.setAlpha(255 if mode == "cover" else int(max(0.0, min(1.0, opacity)) * 255))

        block = rect.adjusted(
            int(padding_x), int(padding_y), -int(padding_x), -int(padding_y)
        )

        path = QPainterPath()
        path.addRoundedRect(
            float(block.x()), float(block.y()),
            float(block.width()), float(block.height()), radius, radius
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(base)
        painter.drawPath(path)

        bold = bool(getattr(style, "bold", False))

        text_color = QColor(getattr(style, "text_color", "#FFFFFF"))
        if not text_color.isValid():
            text_color = QColor("#FFFFFF")

        painter.setFont(base_font(self._theme, "body", "semibold" if bold else "regular"))
        painter.setPen(text_color)
        painter.drawText(
            block,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            text,
        )


#: 开源组件清单，对应《项目原创性声明》三、开源组件引用清单。
_OPEN_SOURCE_COMPONENTS: tuple[tuple[str, str, str], ...] = (
    ("python-mss", "跨平台屏幕实时捕获", "MIT License"),
    ("RapidOCR", "文字检测与识别", "Apache-2.0 License"),
    ("OpenCV-Python", "图像处理与画面差分", "Apache-2.0 License"),
    ("Spark-X2.5 系列", "本地翻译大模型", "Apache-2.0 License"),
    ("llama.cpp", "GGUF 量化与轻量化推理", "MIT License"),
    ("PyQt6", "GUI 与透明渲染层", "GPL v3 / 商业双协议"),
    ("psutil", "系统资源监控", "BSD 3-Clause License"),
)
