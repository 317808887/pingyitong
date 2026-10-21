"""支持语言表。

本模块是语言相关信息的唯一来源：语言代码、中文名称、英文名称与文字脚本。

与模型能力的关系（如实说明）
    Spark-X2.5-1.7B / 4B 的官方说明是 "support for more than 200 languages"，
    但**官方并未公布这 200+ 种语言的具体清单**（已核对模型仓库 README 与
    config.json，其中只有这一句总体声明）。

    因此本项目采取两条路：

    1. :data:`LANGUAGES` 是一份可选语言清单，用于界面选择；
       它不等于官方清单，也不保证对应 OCR 或每个翻译方向可用；
    2. :data:`SUPPORTED_VERIFIED` 由 ``scripts/test_languages.py`` **实测**
       得到：用固定探针句逐语言试译，记录输出文字脚本是否符合目标。
       同一脚本可能对应多种语言；这些历史标记不证明语言身份或翻译准确度。

    :func:`is_verified` 可查询某个语言是否通过了实测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class LanguageInfo:
    """一种语言的元信息。"""

    #: ISO 639-1 代码（少数语言无两位代码时用 ISO 639-3）
    code: str
    #: 中文名称（界面展示）
    name: str
    #: 英文名称（构造 Prompt 时使用，模型对英文语言名最可靠）
    english_name: str
    #: 该语言母语者书写的名称，用于强化 Prompt 指向
    native_name: str = ""
    #: 主要文字脚本，用于输出脚本校验，不能单独证明语言身份
    script: str = "latin"

    def prompt_name(self) -> str:
        """返回用于 Prompt 的语言名称。

        同时给出英文名与母语名，能显著提高模型对冷门语言的命中率——
        只给英文名时，模型容易把冷门语言当成近亲语言处理。
        """
        if self.native_name and self.native_name != self.english_name:
            return f"{self.english_name} ({self.native_name})"
        return self.english_name


# ----------------------------------------------------------------------
# 界面可选语言清单
# ----------------------------------------------------------------------
#
# 组织顺序：常用语言在前，其余按中文名称排序，便于在长列表中查找。

#: 最常用的一批（界面置顶，避免用户在下拉里翻找）
_COMMON: Final[tuple[tuple[str, str, str, str, str], ...]] = (
    # code, 中文名, 英文名, 母语名, 脚本
    ("zh", "中文（简体）", "Simplified Chinese", "简体中文", "han"),
    ("zh-TW", "中文（繁体）", "Traditional Chinese", "繁體中文", "han"),
    ("en", "英语", "English", "English", "latin"),
    ("ja", "日语", "Japanese", "日本語", "japanese"),
    ("ko", "韩语", "Korean", "한국어", "hangul"),
    ("fr", "法语", "French", "Français", "latin"),
    ("de", "德语", "German", "Deutsch", "latin"),
    ("es", "西班牙语", "Spanish", "Español", "latin"),
    ("ru", "俄语", "Russian", "Русский", "cyrillic"),
    ("pt", "葡萄牙语", "Portuguese", "Português", "latin"),
    ("it", "意大利语", "Italian", "Italiano", "latin"),
    ("ar", "阿拉伯语", "Arabic", "العربية", "arabic"),
)

#: 其余语言（含 Edge/Chrome 翻译插件提供的语种，以及主要区域语言）
_EXTENDED: Final[tuple[tuple[str, str, str, str, str], ...]] = (
    ("af", "南非荷兰语", "Afrikaans", "Afrikaans", "latin"),
    ("sq", "阿尔巴尼亚语", "Albanian", "Shqip", "latin"),
    ("am", "阿姆哈拉语", "Amharic", "አማርኛ", "ethiopic"),
    ("hy", "亚美尼亚语", "Armenian", "Հայերեն", "armenian"),
    ("as", "阿萨姆语", "Assamese", "অসমীয়া", "bengali"),
    ("az", "阿塞拜疆语", "Azerbaijani", "Azərbaycan", "latin"),
    ("eu", "巴斯克语", "Basque", "Euskara", "latin"),
    ("be", "白俄罗斯语", "Belarusian", "Беларуская", "cyrillic"),
    ("bn", "孟加拉语", "Bengali", "বাংলা", "bengali"),
    ("bs", "波斯尼亚语", "Bosnian", "Bosanski", "latin"),
    ("bg", "保加利亚语", "Bulgarian", "Български", "cyrillic"),
    ("my", "缅甸语", "Burmese", "မြန်မာ", "myanmar"),
    ("ca", "加泰罗尼亚语", "Catalan", "Català", "latin"),
    ("ceb", "宿务语", "Cebuano", "Cebuano", "latin"),
    ("ny", "齐切瓦语", "Chichewa", "Chichewa", "latin"),
    ("co", "科西嘉语", "Corsican", "Corsu", "latin"),
    ("hr", "克罗地亚语", "Croatian", "Hrvatski", "latin"),
    ("cs", "捷克语", "Czech", "Čeština", "latin"),
    ("da", "丹麦语", "Danish", "Dansk", "latin"),
    ("nl", "荷兰语", "Dutch", "Nederlands", "latin"),
    ("eo", "世界语", "Esperanto", "Esperanto", "latin"),
    ("et", "爱沙尼亚语", "Estonian", "Eesti", "latin"),
    ("tl", "菲律宾语", "Filipino", "Filipino", "latin"),
    ("fi", "芬兰语", "Finnish", "Suomi", "latin"),
    ("fy", "弗里西语", "Frisian", "Frysk", "latin"),
    ("gl", "加利西亚语", "Galician", "Galego", "latin"),
    ("ka", "格鲁吉亚语", "Georgian", "ქართული", "georgian"),
    ("el", "希腊语", "Greek", "Ελληνικά", "greek"),
    ("gu", "古吉拉特语", "Gujarati", "ગુજરાતી", "gujarati"),
    ("ht", "海地克里奥尔语", "Haitian Creole", "Kreyòl ayisyen", "latin"),
    ("ha", "豪萨语", "Hausa", "Hausa", "latin"),
    ("haw", "夏威夷语", "Hawaiian", "ʻŌlelo Hawaiʻi", "latin"),
    ("he", "希伯来语", "Hebrew", "עברית", "hebrew"),
    ("hi", "印地语", "Hindi", "हिन्दी", "devanagari"),
    ("hmn", "苗语", "Hmong", "Hmoob", "latin"),
    ("hu", "匈牙利语", "Hungarian", "Magyar", "latin"),
    ("is", "冰岛语", "Icelandic", "Íslenska", "latin"),
    ("ig", "伊博语", "Igbo", "Igbo", "latin"),
    ("id", "印尼语", "Indonesian", "Bahasa Indonesia", "latin"),
    ("ga", "爱尔兰语", "Irish", "Gaeilge", "latin"),
    ("jw", "爪哇语", "Javanese", "Basa Jawa", "latin"),
    ("kn", "卡纳达语", "Kannada", "ಕನ್ನಡ", "kannada"),
    ("kk", "哈萨克语", "Kazakh", "Қазақ тілі", "cyrillic"),
    ("km", "高棉语", "Khmer", "ខ្មែរ", "khmer"),
    ("rw", "卢旺达语", "Kinyarwanda", "Ikinyarwanda", "latin"),
    ("ku", "库尔德语", "Kurdish", "Kurdî", "latin"),
    ("ky", "吉尔吉斯语", "Kyrgyz", "Кыргызча", "cyrillic"),
    ("lo", "老挝语", "Lao", "ລາວ", "lao"),
    ("la", "拉丁语", "Latin", "Latina", "latin"),
    ("lv", "拉脱维亚语", "Latvian", "Latviešu", "latin"),
    ("lt", "立陶宛语", "Lithuanian", "Lietuvių", "latin"),
    ("lb", "卢森堡语", "Luxembourgish", "Lëtzebuergesch", "latin"),
    ("mk", "马其顿语", "Macedonian", "Македонски", "cyrillic"),
    ("mg", "马尔加什语", "Malagasy", "Malagasy", "latin"),
    ("ms", "马来语", "Malay", "Bahasa Melayu", "latin"),
    ("ml", "马拉雅拉姆语", "Malayalam", "മലയാളം", "malayalam"),
    ("mt", "马耳他语", "Maltese", "Malti", "latin"),
    ("mi", "毛利语", "Maori", "Te Reo Māori", "latin"),
    ("mr", "马拉地语", "Marathi", "मराठी", "devanagari"),
    ("mn", "蒙古语", "Mongolian", "Монгол", "cyrillic"),
    ("ne", "尼泊尔语", "Nepali", "नेपाली", "devanagari"),
    ("no", "挪威语", "Norwegian", "Norsk", "latin"),
    ("or", "奥里亚语", "Odia", "ଓଡ଼ିଆ", "oriya"),
    ("ps", "普什图语", "Pashto", "پښتو", "arabic"),
    ("fa", "波斯语", "Persian", "فارسی", "arabic"),
    ("pl", "波兰语", "Polish", "Polski", "latin"),
    ("pa", "旁遮普语", "Punjabi", "ਪੰਜਾਬੀ", "gurmukhi"),
    ("ro", "罗马尼亚语", "Romanian", "Română", "latin"),
    ("sm", "萨摩亚语", "Samoan", "Gagana Samoa", "latin"),
    ("gd", "苏格兰盖尔语", "Scots Gaelic", "Gàidhlig", "latin"),
    ("sr", "塞尔维亚语", "Serbian", "Српски", "cyrillic"),
    ("st", "塞索托语", "Sesotho", "Sesotho", "latin"),
    ("sn", "绍纳语", "Shona", "ChiShona", "latin"),
    ("sd", "信德语", "Sindhi", "سنڌي", "arabic"),
    ("si", "僧伽罗语", "Sinhala", "සිංහල", "sinhala"),
    ("sk", "斯洛伐克语", "Slovak", "Slovenčina", "latin"),
    ("sl", "斯洛文尼亚语", "Slovenian", "Slovenščina", "latin"),
    ("so", "索马里语", "Somali", "Soomaali", "latin"),
    ("su", "巽他语", "Sundanese", "Basa Sunda", "latin"),
    ("sw", "斯瓦希里语", "Swahili", "Kiswahili", "latin"),
    ("sv", "瑞典语", "Swedish", "Svenska", "latin"),
    ("tg", "塔吉克语", "Tajik", "Тоҷикӣ", "cyrillic"),
    ("ta", "泰米尔语", "Tamil", "தமிழ்", "tamil"),
    ("tt", "鞑靼语", "Tatar", "Татарча", "cyrillic"),
    ("te", "泰卢固语", "Telugu", "తెలుగు", "telugu"),
    ("th", "泰语", "Thai", "ไทย", "thai"),
    ("tr", "土耳其语", "Turkish", "Türkçe", "latin"),
    ("tk", "土库曼语", "Turkmen", "Türkmençe", "latin"),
    ("uk", "乌克兰语", "Ukrainian", "Українська", "cyrillic"),
    ("ur", "乌尔都语", "Urdu", "اردو", "arabic"),
    ("ug", "维吾尔语", "Uyghur", "ئۇيغۇرچە", "arabic"),
    ("uz", "乌兹别克语", "Uzbek", "Oʻzbekcha", "latin"),
    ("vi", "越南语", "Vietnamese", "Tiếng Việt", "latin"),
    ("cy", "威尔士语", "Welsh", "Cymraeg", "latin"),
    ("xh", "科萨语", "Xhosa", "isiXhosa", "latin"),
    ("yi", "意第绪语", "Yiddish", "ייִדיש", "hebrew"),
    ("yo", "约鲁巴语", "Yoruba", "Yorùbá", "latin"),
    ("zu", "祖鲁语", "Zulu", "isiZulu", "latin"),
    ("bo", "藏语", "Tibetan", "བོད་སྐད་", "tibetan"),
    ("dv", "迪维希语", "Dhivehi", "ދިވެހި", "thaana"),
    ("ti", "提格里尼亚语", "Tigrinya", "ትግርኛ", "ethiopic"),
    ("om", "奥罗莫语", "Oromo", "Afaan Oromoo", "latin"),
    ("qu", "克丘亚语", "Quechua", "Runasimi", "latin"),
    ("gn", "瓜拉尼语", "Guarani", "Avañe'ẽ", "latin"),
    ("ay", "艾马拉语", "Aymara", "Aymar aru", "latin"),
    ("nso", "北索托语", "Northern Sotho", "Sesotho sa Leboa", "latin"),
    ("tn", "茨瓦纳语", "Tswana", "Setswana", "latin"),
    ("ts", "聪加语", "Tsonga", "Xitsonga", "latin"),
    ("ss", "斯瓦蒂语", "Swati", "SiSwati", "latin"),
    ("ve", "文达语", "Venda", "Tshivenḓa", "latin"),
    ("bm", "班巴拉语", "Bambara", "Bamanankan", "latin"),
    ("wo", "沃洛夫语", "Wolof", "Wolof", "latin"),
    ("ff", "富拉语", "Fulah", "Fulfulde", "latin"),
    ("ln", "林加拉语", "Lingala", "Lingála", "latin"),
    ("lg", "卢干达语", "Luganda", "Luganda", "latin"),
    ("ak", "阿坎语", "Akan", "Akan", "latin"),
    ("ee", "埃维语", "Ewe", "Eʋegbe", "latin"),
    ("tw", "契维语", "Twi", "Twi", "latin"),
    ("br", "布列塔尼语", "Breton", "Brezhoneg", "latin"),
    ("oc", "奥克语", "Occitan", "Occitan", "latin"),
    ("sc", "撒丁语", "Sardinian", "Sardu", "latin"),
    ("rm", "罗曼什语", "Romansh", "Rumantsch", "latin"),
    ("fo", "法罗语", "Faroese", "Føroyskt", "latin"),
    ("kl", "格陵兰语", "Kalaallisut", "Kalaallisut", "latin"),
    ("iu", "因纽特语", "Inuktitut", "ᐃᓄᒃᑎᑐᑦ", "inuktitut"),
    ("chr", "切罗基语", "Cherokee", "ᏣᎳᎩ", "cherokee"),
    ("nv", "纳瓦霍语", "Navajo", "Diné bizaad", "latin"),
    ("oj", "奥吉布瓦语", "Ojibwe", "Anishinaabemowin", "latin"),
    ("cr", "克里语", "Cree", "ᓀᐦᐃᔭᐍᐏᐣ", "cree"),
    ("moh", "莫霍克语", "Mohawk", "Kanienʼkéha", "latin"),
)

#: 其余 ISO 639-1 语言与主要 ISO 639-3 语言。
#:
#: 为什么要收这么全：模型的官方说法是"支持 200+ 种语言"，
#: 而候选清单必须覆盖到同一量级，实测结论才有意义——
#: 只在十几种语言上验证，无法支撑或证伪"200+"这个说法。
#: 本组以 ISO 639-1 的完整集合为主，并补充使用人口较多的 ISO 639-3 语言。
_LONG_TAIL: Final[tuple[tuple[str, str, str, str, str], ...]] = (
    ("aa", "阿法尔语", "Afar", "Afaraf", "latin"),
    ("ab", "阿布哈兹语", "Abkhazian", "Аҧсуа", "cyrillic"),
    ("ae", "阿维斯陀语", "Avestan", "Avesta", "arabic"),
    ("an", "阿拉贡语", "Aragonese", "Aragonés", "latin"),
    ("av", "阿瓦尔语", "Avaric", "Авар", "cyrillic"),
    ("ba", "巴什基尔语", "Bashkir", "Башҡортса", "cyrillic"),
    ("bho", "博杰普尔语", "Bhojpuri", "भोजपुरी", "devanagari"),
    ("bi", "比斯拉马语", "Bislama", "Bislama", "latin"),
    ("ch", "查莫罗语", "Chamorro", "Chamoru", "latin"),
    ("cv", "楚瓦什语", "Chuvash", "Чӑвашла", "cyrillic"),
    ("fj", "斐济语", "Fijian", "Na Vosa Vakaviti", "latin"),
    ("gv", "曼岛语", "Manx", "Gaelg", "latin"),
    ("ho", "希里莫图语", "Hiri Motu", "Hiri Motu", "latin"),
    ("hz", "赫雷罗语", "Herero", "Otjiherero", "latin"),
    ("ia", "国际语", "Interlingua", "Interlingua", "latin"),
    ("ie", "西方国际语", "Interlingue", "Interlingue", "latin"),
    ("ii", "彝语", "Sichuan Yi", "ꆈꌠꉙ", "yi"),
    ("io", "伊多语", "Ido", "Ido", "latin"),
    ("kg", "刚果语", "Kongo", "Kikongo", "latin"),
    ("ki", "基库尤语", "Kikuyu", "Gĩkũyũ", "latin"),
    ("kj", "宽亚玛语", "Kuanyama", "Oshikwanyama", "latin"),
    ("ks", "克什米尔语", "Kashmiri", "کٲشُر", "arabic"),
    ("kv", "科米语", "Komi", "Коми кыв", "cyrillic"),
    ("kw", "康沃尔语", "Cornish", "Kernewek", "latin"),
    ("li", "林堡语", "Limburgish", "Limburgs", "latin"),
    ("lu", "卢巴加丹加语", "Luba-Katanga", "Kiluba", "latin"),
    ("mh", "马绍尔语", "Marshallese", "Kajin M̧ajeļ", "latin"),
    ("na", "瑙鲁语", "Nauru", "Dorerin Naoero", "latin"),
    ("nb", "书面挪威语", "Norwegian Bokmål", "Norsk bokmål", "latin"),
    ("nd", "北恩德贝莱语", "North Ndebele", "isiNdebele", "latin"),
    ("ng", "恩敦加语", "Ndonga", "Oshindonga", "latin"),
    ("nn", "新挪威语", "Norwegian Nynorsk", "Nynorsk", "latin"),
    ("nr", "南恩德贝莱语", "South Ndebele", "isiNdebele", "latin"),
    ("os", "奥塞梯语", "Ossetian", "Ирон æвзаг", "cyrillic"),
    ("pi", "巴利语", "Pali", "पालि", "devanagari"),
    ("rn", "基隆迪语", "Rundi", "Ikirundi", "latin"),
    ("sa", "梵语", "Sanskrit", "संस्कृतम्", "devanagari"),
    ("se", "北萨米语", "Northern Sami", "Davvisámegiella", "latin"),
    ("sg", "桑戈语", "Sango", "Sängö", "latin"),
    ("to", "汤加语", "Tongan", "Lea faka-Tonga", "latin"),
    ("ty", "塔希提语", "Tahitian", "Reo Tahiti", "latin"),
    ("vo", "沃拉普克语", "Volapük", "Volapük", "latin"),
    ("wa", "瓦隆语", "Walloon", "Walon", "latin"),
    ("za", "壮语", "Zhuang", "Vahcuengh", "latin"),
    ("ilo", "伊洛卡诺语", "Ilocano", "Ilokano", "latin"),
    ("war", "瓦瑞语", "Waray", "Winaray", "latin"),
    ("pam", "邦板牙语", "Kapampangan", "Kapampangan", "latin"),
    ("bik", "比科尔语", "Bikol", "Bikol", "latin"),
    ("hil", "希利盖农语", "Hiligaynon", "Hiligaynon", "latin"),
    ("min", "米南加保语", "Minangkabau", "Minangkabau", "latin"),
    ("ban", "巴厘语", "Balinese", "Basa Bali", "latin"),
    ("bug", "布吉语", "Buginese", "ᨅᨔ ᨕᨘᨁᨗ", "lontara"),
    ("syr", "叙利亚语", "Syriac", "ܣܘܪܝܝܐ", "syriac"),
    ("arc", "亚拉姆语", "Aramaic", "ארמית", "hebrew"),
    ("cop", "科普特语", "Coptic", "ⲘⲉⲧⲢⲉⲙⲛ̀ⲭⲏⲙⲓ", "coptic"),
    ("grc", "古希腊语", "Ancient Greek", "Ἀρχαία Ἑλληνικὴ", "greek"),
    ("yue", "粤语", "Cantonese", "粵語", "han"),
    ("wuu", "吴语", "Wu Chinese", "吴语", "han"),
    ("nan", "闽南语", "Min Nan Chinese", "閩南語", "han"),
    ("hak", "客家语", "Hakka Chinese", "客家話", "han"),
    ("gom", "孔卡尼语", "Konkani", "कोंकणी", "devanagari"),
    ("mai", "迈蒂利语", "Maithili", "मैथिली", "devanagari"),
    ("sat", "桑塔利语", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "ol_chiki"),
    ("new", "尼瓦尔语", "Newari", "नेपाल भाषा", "devanagari"),
    ("ace", "亚齐语", "Acehnese", "Bahsa Acèh", "latin"),
    ("bew", "巴达维语", "Betawi", "Bahasa Betawi", "latin"),
    ("mad", "马都拉语", "Madurese", "Madhurâ", "latin"),
    ("tsg", "陶苏格语", "Tausug", "Bahasa Sūg", "latin"),
    ("tet", "德顿语", "Tetum", "Tetun", "latin"),
    ("dz", "宗喀语", "Dzongkha", "རྫོང་ཁ", "tibetan"),
    ("bal", "俾路支语", "Balochi", "بلوچی", "arabic"),
    ("brh", "布拉灰语", "Brahui", "براہوئی", "arabic"),
    ("lah", "拉亨达语", "Lahnda", "لہندا", "arabic"),
    ("skr", "色莱基语", "Saraiki", "سرائیکی", "arabic"),
)


def _build_languages() -> dict[str, LanguageInfo]:
    """构造语言代码到语言信息的映射，并保持插入顺序（常用语在前）。"""
    result: dict[str, LanguageInfo] = {}
    for code, name, english, native, script in _COMMON + _EXTENDED:
        # 重复代码（如误加的 fy）以首次出现为准，避免覆盖
        result.setdefault(
            code, LanguageInfo(code, name, english, native, script)
        )
    return result


#: 全部可选语言，键为语言代码。插入顺序即界面展示顺序（常用语在前）。
LANGUAGES: Final[dict[str, LanguageInfo]] = _build_languages()

# ----------------------------------------------------------------------
# 实测支持档位
# ----------------------------------------------------------------------
#
# 实测结论见 language_test_results（由 scripts/test_languages.py 生成），
# 这里只做转发，避免把实验数据与语言定义混在一起。
#
# 注意导入位置：放在文件末尾是为了避免与 language_test_results 形成循环导入
# （后者在文档字符串里引用了本模块）。

from .language_test_results import (  # noqa: E402 - 置于文件末尾以避免循环导入
    FULLY_TESTED as _FULLY_TESTED,
)
from .language_test_results import (  # noqa: E402
    VERIFIED_BOTH as _VERIFIED_BOTH,
)
from .language_test_results import (  # noqa: E402
    is_verified as _is_verified,
)
from .language_test_results import (  # noqa: E402
    status as _status,
)
from .language_test_results import (  # noqa: E402
    summary as _summary,
)


def verification_status(code: str) -> str:
    """返回语言在实测中的支持档位。

    :return: ``"verified"``（实测确定支持）/ ``"likely"``（疑似支持）/
        ``"unsupported"``（实测不支持）。
    """
    return _status(code)


def is_verified(code: str) -> bool:
    """语言是否实测确定支持。"""
    return _is_verified(code)


def test_summary() -> str:
    """返回实测结论摘要（界面展示用）。

    传入候选集才能给出"不支持"的数量——它由补集推出，
    实测结果模块本身不知道候选集有多大。
    """
    return _summary(frozenset(LANGUAGES))


def is_fully_tested() -> bool:
    """双模型是否都已完成实测。"""
    return _FULLY_TESTED


def verified_codes() -> frozenset[str]:
    """返回实测确定支持的语言代码集合。"""
    return _VERIFIED_BOTH
