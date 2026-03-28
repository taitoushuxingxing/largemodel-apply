"""
新闻分类原始数据生成脚本
生成约 10,000 条数据，包含 query, intent 两列
5 个类别：sports, technology, entertainment, finance, military
"""

import csv
import random
import os
from collections import Counter

random.seed(42)

# ============================================================
# 1. 定义 5 个意图类别及模板 + 词槽
# ============================================================

INTENT_TEMPLATES = {

    # -------------------------------------------------------
    # 体育 (sports)
    # -------------------------------------------------------
    "sports": {
        "templates": [
            "{team}以{score}击败{team2}，拿下{tournament}冠军",
            "{player}在{tournament}中打入{num}球，助{team}取胜",
            "{player}宣布从{sport}退役，职业生涯共获{num}座冠军",
            "{tournament}{year}赛季正式开幕，{num}支球队参赛",
            "{player}转会{team}，转会费高达{money}",
            "{team}主教练{coach}因战绩不佳被解雇",
            "{player}在比赛中受伤，预计缺阵{num}周",
            "{country}队在{tournament}中爆冷击败{country2}队",
            "{sport}世界纪录被打破，{player}创造新历史",
            "{team}连续{num}场不胜，球迷呼吁换帅",
            "{player}荣获{year}年度{award}",
            "{tournament}决赛将于{month}在{city}举行",
            "{team}官宣签下{country}球星{player}",
            "{sport}协会公布新赛季规则调整方案",
            "{player}复出首秀表现亮眼，贡献{num}分",
            "{team}青训小将{player}一战成名",
            "{tournament}半决赛：{team}对阵{team2}",
            "{country}申办{year}年{tournament}成功",
            "{player}因违规被禁赛{num}场",
            "{sport}联赛最新积分榜：{team}领跑",
            "伤病潮来袭，{team}多名主力缺阵{tournament}",
            "{player}赛后采访表示会继续努力拿到冠军",
            "{team}发布新赛季球衣，设计灵感来源于{city}",
            "{tournament}抽签结果出炉，{team}遭遇{team2}",
            "{player}社交媒体粉丝突破{num}万",
            "{sport}运动在{country}越来越受欢迎",
            "{team}主场迎���{team2}，门票{num}分钟售罄",
            "{player}带伤上阵，{team}惜败{team2}",
            "{coach}执教{team}首秀告捷",
            "{tournament}最佳阵容公布，{player}入选",
        ],
        "team": ["湖人", "勇士", "皇马", "巴萨", "曼联", "利物浦", "拜仁", "国际米兰",
                 "北京国安", "上海申花", "广州恒大", "山东泰山", "切尔西", "巴黎圣日耳曼",
                 "尤文图斯", "多特蒙德", "热刺", "阿森纳", "AC米兰", "那不勒斯"],
        "team2": ["湖人", "勇士", "皇马", "巴萨", "曼联", "利物浦", "拜仁", "国际米兰",
                  "切尔西", "巴黎圣日耳曼", "尤文图斯", "多特蒙德", "热刺", "阿森纳"],
        "player": ["梅西", "C罗", "姆巴佩", "哈兰德", "詹姆斯", "库里", "武磊", "孙兴慜",
                   "德约科维奇", "纳达尔", "谷爱凌", "苏炳添", "全红婵", "郑钦文",
                   "字母哥", "东契奇", "恩比德", "贝林厄姆", "维尼修斯", "萨拉赫"],
        "coach": ["瓜迪奥拉", "安切洛蒂", "克洛普", "穆里尼奥", "滕哈赫", "波波维奇",
                  "里皮", "哈维", "图赫尔", "阿尔特塔"],
        "tournament": ["世界杯", "欧冠", "英超", "西甲", "NBA", "中超", "奥运会",
                       "亚洲杯", "欧洲杯", "美洲杯", "法网", "温网", "澳网"],
        "sport": ["足球", "篮球", "网球", "乒乓球", "游泳", "田径", "羽毛球", "排球",
                  "滑雪", "短道速滑", "拳击", "高尔夫"],
        "country": ["中国", "巴西", "阿根廷", "法国", "德国", "西班牙", "英格兰", "日本", "韩国", "美国"],
        "country2": ["中国", "巴西", "阿根廷", "法国", "德国", "西班牙", "英格兰", "日本", "韩国", "美国"],
        "city": ["北京", "上海", "伦敦", "巴黎", "东京", "纽约", "柏林", "马德里", "巴塞罗那", "多哈"],
        "score": ["3:1", "2:0", "4:2", "1:0", "3:2", "5:3", "2:1", "110:98", "120:105", "98:87"],
        "num": ["2", "3", "5", "7", "10", "15", "20", "25", "30", "50", "100", "500"],
        "money": ["5000万欧元", "1亿欧元", "8000万美元", "2亿美元", "3000万英镑", "1.5亿欧元"],
        "year": ["2024", "2025", "2026", "2028", "2030"],
        "month": ["1月", "3月", "5月", "6月", "7月", "8月", "10月", "12月"],
        "award": ["金球奖", "最佳球员", "MVP", "金靴奖", "最佳新秀", "最佳教练"],
    },

    # -------------------------------------------------------
    # 科技 (technology)
    # -------------------------------------------------------
    "technology": {
        "templates": [
            "{company}发布{product}，搭载{tech}技术",
            "{company}最新{product}正式开售，售价{price}元起",
            "{company}宣布投资{money}研发{tech}",
            "{tech}技术取得突破，{company}率先实现商用",
            "{company}CEO{person}宣布{product}将于{month}发布",
            "{product}评测：{tech}性能提升{num}%",
            "{company}与{company2}达成{tech}领域战略合作",
            "{company}申请{num}项{tech}相关专利",
            "{tech}行业报告：市场规模将在{year}年达到{money}",
            "{company}开源了其{tech}框架，引发开发者热议",
            "全球{tech}大会在{city}开幕，{company}发表主题演讲",
            "{company}{product}销量突破{num}万台",
            "{person}预测{tech}将在{num}年内改变{industry}",
            "{company}裁员{num}%，将资源集中到{tech}业务",
            "{product}系统更新：新增{feature}功能",
            "{company}发布{year}年技术趋势白皮书",
            "{tech}芯片性能跑分曝光，超越{company2}旗舰",
            "{company}自研{tech}芯片流片成功",
            "{country}{tech}产业政策出台，扶持力度加大",
            "研究人员利用{tech}成功{achievement}",
            "{company}推出{tech}开发者平台",
            "{product}被曝存在{issue}问题，{company}回应将尽快修复",
            "{company}获得{money}融资，估值达{valuation}",
            "{tech}领域最新论文：{achievement}",
            "{company}在{city}设立{tech}研究院",
            "{person}加入{company}，负责{tech}业务",
            "{company}发布{tech}大模型，参数量达{num}亿",
            "{product}国行版正式发售，支持{feature}",
            "{tech}安全漏洞被发现，影响{num}万用户",
            "IDC报告：{company}在{tech}市场份额排名第一",
        ],
        "company": ["华为", "苹果", "谷歌", "微软", "小米", "OPPO", "vivo", "三星",
                    "特斯拉", "英伟达", "台积电", "高通", "联想", "百度", "阿里", "腾讯",
                    "字节跳动", "OpenAI", "Meta", "亚马逊", "比亚迪", "大疆", "荣耀"],
        "company2": ["华为", "苹果", "谷歌", "微软", "小米", "三星", "英伟达", "高通",
                     "百度", "阿里", "腾讯", "OpenAI", "Meta", "AMD", "英特尔"],
        "product": ["手机", "平板", "笔记本", "智能手表", "耳机", "电视", "芯片", "操作系统",
                    "无人机", "机器人", "智能音箱", "VR头显", "AR眼镜", "折叠屏手机",
                    "智能汽车", "AI助手", "大模型", "搜索引擎"],
        "tech": ["AI", "5G", "6G", "量子计算", "区块链", "自动驾驶", "大模型", "芯片",
                "物联网", "云计算", "边缘计算", "AIGC", "机器人", "光刻", "卫星通信",
                "混合现实", "脑机接口", "基因编辑", "固态电池", "氢能源"],
        "person": ["马斯克", "黄仁勋", "雷军", "任正非", "库克", "Sam Altman", "扎克伯格",
                   "纳德拉", "李彦宏", "张一鸣", "余承东", "马化腾"],
        "price": ["1999", "2999", "3999", "4999", "5999", "6999", "7999", "9999", "12999"],
        "money": ["10亿", "50亿", "100亿", "200亿", "500亿", "1000亿"],
        "valuation": ["100亿美元", "500亿美元", "1000亿美元", "万亿美元"],
        "num": ["5", "10", "15", "20", "30", "50", "100", "200", "500", "1000", "7000"],
        "year": ["2025", "2026", "2027", "2028", "2030"],
        "month": ["1月", "3月", "5月", "6月", "8月", "9月", "10月", "11月"],
        "city": ["北京", "上海", "深圳", "杭州", "旧金山", "拉斯维加斯", "巴塞罗那", "东京", "西雅图"],
        "country": ["中国", "美国", "欧盟", "日本", "韩国", "印度"],
        "industry": ["医疗", "教育", "制造业", "金融", "交通", "农业", "零售"],
        "feature": ["AI翻译", "离线模式", "卫星通话", "手势控制", "多模态交互", "隔空投送"],
        "issue": ["发热", "续航不足", "信号不稳定", "屏幕闪烁", "系统卡顿", "数据泄露"],
        "achievement": ["将推理速度提升10倍", "实现了零样本学习突破", "在多个基准测试中刷新纪录",
                        "首次在真实环境中实现全自动操作", "将模型体积压缩90%"],
    },

    # -------------------------------------------------------
    # 娱乐 (entertainment)
    # -------------------------------------------------------
    "entertainment": {
        "templates": [
            "{star}新片《{movie}》定档{month}，预告片曝光",
            "电影《{movie}》票房突破{num}亿，创{genre}片新纪录",
            "{star}官宣出演{director}导演新作",
            "综艺《{show}》第{season}季回归，{star}加盟",
            "{star}在{event}上获得{award}",
            "{star}与{star2}合作新歌《{song}》上线",
            "电视剧《{drama}》热播，豆瓣评分高达{score}",
            "{star}演唱会{city}站门票{num}秒售罄",
            "{director}新片入围{festival}主竞赛单元",
            "《{movie}》发布终极预告，特效震撼",
            "{star}深夜发文回应{controversy}争议",
            "{platform}独播剧《{drama}》口碑爆棚",
            "《{drama}》大结局收视率破{num}",
            "{star}婚礼现场曝光，众多好友到场祝贺",
            "{star}登上{magazine}杂志封面",
            "《{movie}》导演{director}谈创作幕后故事",
            "{star}直播带货首秀，{num}分钟成交额破亿",
            "金鸡奖提名公布：《{movie}》获{num}项提名",
            "{star}主演的《{drama}》正式杀青",
            "《{show}》最新一期收视登顶，{star}表现抢眼",
            "{star}宣布暂别娱乐圈，将专注个人生活",
            "动画电影《{movie}》口碑逆袭，票房持续走高",
            "{star2}翻唱{star}经典歌曲走红网络",
            "《{drama}》选角争议：网友热议{star}是否适合角色",
            "{event}红毯造型盘点：{star}惊艳亮相",
            "剧版《{movie}》确认开拍，{star}将出演主角",
            "{star}新专辑《{song}》首周销量破{num}万",
            "{platform}公布{year}年度最受欢迎影视榜单",
            "《{show}》嘉宾阵容官宣：{star}、{star2}等加盟",
            "{star}被曝将参演好莱坞大片",
        ],
        "star": ["易烊千玺", "赵丽颖", "王一博", "杨幂", "朱一龙", "迪丽热巴", "肖战",
                 "刘亦菲", "吴京", "沈腾", "章子怡", "黄渤", "周迅", "邓超", "杨紫",
                 "龚俊", "白鹿", "成毅", "刘诗诗", "彭于晏", "张译", "雷佳音"],
        "star2": ["易烊千玺", "赵丽颖", "王一博", "杨幂", "朱一龙", "迪丽热巴",
                  "刘亦菲", "吴京", "沈腾", "黄渤", "杨紫", "白鹿", "张译"],
        "movie": ["流浪地球3", "封神2", "长津湖", "满江红", "热辣滚烫", "飞驰人生3",
                  "唐探4", "战狼3", "哪吒2", "消失的她2", "八角笼中", "孤注一掷",
                  "三体", "深海2", "熊出没大电影", "功夫熊猫5"],
        "drama": ["庆余年3", "狂飙2", "繁花", "长相思", "莲花楼", "玫瑰的故事",
                  "墨雨云间", "与凤行", "度华年", "苍兰诀2", "知否2", "甄嬛传前传"],
        "show": ["跑男", "歌手", "向往的生活", "极限挑战", "中国好声音", "脱口秀大会",
                "乘风破浪", "披荆斩棘", "声生不息", "大侦探"],
        "director": ["张艺谋", "陈凯歌", "贾樟柯", "王家卫", "郭帆", "乌尔善", "大鹏",
                     "韩寒", "徐克", "陈思诚", "文牧野", "曹保平"],
        "song": ["孤勇者", "花海", "错位时空", "如愿", "漠河舞厅", "起风了", "年少有为",
                "大鱼", "光年之外", "晚风心里吹"],
        "genre": ["科幻", "喜剧", "动作", "悬疑", "爱情", "动画", "战争", "奇幻"],
        "festival": ["戛纳电影节", "威尼斯电影���", "柏林电影节", "东京电影节", "金鸡奖", "金马奖"],
        "event": ["金鸡奖", "百花奖", "金像奖", "微博之夜", "芭莎慈善夜", "跨年晚会"],
        "award": ["最佳男主角", "最佳女主角", "最佳导演", "最受欢迎演员", "年度人气奖", "最佳新人"],
        "platform": ["爱奇艺", "腾讯视频", "优酷", "芒果TV", "B站", "Netflix"],
        "magazine": ["时尚芭莎", "VOGUE", "GQ", "嘉人", "ELLE", "时尚先生"],
        "controversy": ["抄袭", "演技", "恋情", "代言", "片酬", "番位"],
        "city": ["北京", "上海", "广州", "成都", "南京", "杭州", "深圳", "香港", "台北"],
        "num": ["1", "2", "3", "5", "8", "10", "15", "20", "30", "50", "100"],
        "score": ["8.5", "8.8", "9.0", "9.2", "9.5", "7.8", "8.0", "8.3"],
        "season": ["二", "三", "四", "五", "六"],
        "month": ["1月", "2月", "3月", "5月", "7月", "8月", "10月", "12月"],
        "year": ["2025", "2026"],
    },

    # -------------------------------------------------------
    # 财经 (finance)
    # -------------------------------------------------------
    "finance": {
        "templates": [
            "{index}今日{direction}{num}%，{sector}板块领{dir2}",
            "{company}发布{quarter}财报，营收{amount}，同比{direction2}{num}%",
            "央行宣布{policy}，{rate}利率{adj_rate}{num}个基点",
            "{company}股价{direction}{num}%，市值突破{amount}",
            "{currency}汇率创{period}新{high_low}，报{exchange_rate}",
            "{expert}：预计{year}年GDP增速为{num}%",
            "{company}宣布回购{amount}股票",
            "{sector}基金近{period}{direction}{num}%，成市场热点",
            "{country}公布{month}CPI数据，同比{direction2}{num}%",
            "{company}拟{amount}收购{company2}{sector}业务",
            "IPO最新消息：{company}计划在{exchange}上市",
            "{index}突破{num}点关口，创{period}新高",
            "油价{direction}：国内{fuel}价格每升{adj_rate}{num}元",
            "房贷利率再{adj_rate}，首套房利率降至{num}%",
            "{company}被列入{list}，股价应声{direction}",
            "社保基金{quarter}增持{sector}股，持仓曝光",
            "{expert}解读{policy}：对{sector}板块影响几何",
            "多家银行{adj_rate}{deposit}存款利率至{num}%",
            "{company}获{country}政府{amount}补贴",
            "{sector}ETF规模突破{amount}，持续受资金追捧",
            "比特币价格突破{num}美元，{period}涨幅达{pct}%",
            "{company}分红方案公布：每股派{num}元",
            "{country}失业率降至{num}%，就业市场回暖",
            "{index}成交量突破{amount}，市场情绪回暖",
            "外资{period}净买入A股{amount}，看好{sector}板块",
            "{expert}：{sector}行业拐点已到，建议关注龙头",
            "{company}发行{amount}公司债，票面利率{num}%",
            "{month}PMI数据公布：制造业PMI为{pmi}",
            "黄金价格{direction}至每盎司{gold_price}美元",
            "{company}宣布进军{sector}领域，股价大{direction}",
        ],
        "index": ["上证指数", "深证成指", "创业板指", "沪深300", "科创50",
                  "纳斯达克", "道琼斯", "标普500", "恒生指数", "日经225"],
        "company": ["贵州茅台", "宁德时代", "比亚迪", "中国平安", "招商银行",
                    "腾讯控股", "阿里巴巴", "美团", "京东", "拼多多",
                    "中国石油", "工商银行", "中国移动", "中芯国际", "隆基绿能",
                    "苹果", "微软", "英伟达", "亚马逊", "特斯拉"],
        "company2": ["华为", "小米", "字节跳动", "蚂蚁集团", "滴滴", "快手"],
        "sector": ["新能源", "半导体", "AI", "消费", "医药", "银行", "房地产",
                   "军工", "光伏", "锂电", "白酒", "互联网", "汽车", "芯片"],
        "direction": ["涨", "跌", "大涨", "大跌", "暴涨", "暴跌", "微涨", "微跌"],
        "direction2": ["增长", "下降"],
        "dir2": ["涨", "跌"],
        "adj_rate": ["上调", "下调"],
        "num": ["0.5", "1", "1.5", "2", "2.5", "3", "3.5", "5", "8", "10", "15", "20"],
        "pct": ["5", "10", "15", "20", "30", "50", "80"],
        "pmi": ["49.2", "49.8", "50.1", "50.5", "51.0", "51.3", "52.0"],
        "gold_price": ["1950", "2000", "2050", "2100", "2200", "2300", "2400", "2500"],
        "amount": ["10亿", "50亿", "100亿", "200亿", "500亿", "1000亿", "2000亿", "万亿"],
        "currency": ["人民币", "美元", "欧元", "日元", "英镑"],
        "exchange_rate": ["7.10", "7.25", "7.05", "6.95", "1.08", "1.12", "150.5"],
        "rate": ["LPR", "MLF", "存款准备金", "逆回购", "基准"],
        "policy": ["降准", "降息", "加息", "定向降准", "增发国债", "新一轮量化宽松"],
        "expert": ["任泽平", "李迅雷", "管清友", "林毅夫", "高善文", "刘煜辉"],
        "quarter": ["一季度", "二季度", "三季度", "四季度", "上半年", "全年"],
        "period": ["近一周", "近一月", "年内", "历史", "三年"],
        "high_low": ["高", "低"],
        "year": ["2025", "2026", "2027"],
        "month": ["1月", "3月", "5月", "6月", "8月", "10月", "11月", "12月"],
        "country": ["中国", "美国", "欧盟", "日本"],
        "exchange": ["上交所", "深交所", "港交所", "纳斯达克", "纽交所"],
        "fuel": ["汽油", "柴油", "92号汽油", "95号汽油"],
        "list": ["制裁清单", "实体清单", "白名单", "负面清单"],
        "deposit": ["定期", "活期", "大额", "协议"],
    },

    # -------------------------------------------------------
    # 军事 (military)
    # -------------------------------------------------------
    "military": {
        "templates": [
            "{country}军方公布{weapon}最新测试画面",
            "{country}海军{ship}正式服役，排水量达{tonnage}吨",
            "{country}空军{aircraft}完成首飞",
            "{country}与{country2}在{region}举行联合军演",
            "{country}国防预算增至{amount}，同比增长{num}%",
            "{country}成功试射{weapon}，射程覆盖{range}公里",
            "{country}军方宣布在{region}部署{weapon}",
            "{org}发布{year}全球军力排行榜",
            "{country}第{gen}代战斗机项目取得重大进展",
            "{country}与{country2}签署军事合作协议",
            "{country}{branch}举行大规模实弹演习",
            "{country}自主研发的{weapon}系统通过验收",
            "{conflict}最新战况：{war_event}",
            "{country}向{country2}出售{num}架{aircraft}",
            "{leader}视察{country}军事基地",
            "{country}航母{ship}开始海试",
            "{org}峰会讨论{region}安全局势",
            "{country}成功发射{sat_type}卫星",
            "{country}新型{weapon}亮相{mil_event}",
            "{country}军费开支占GDP的{num}%",
            "{country}海军舰队通过{waterway}",
            "分析人士：{country}在{region}的军事存在持续增强",
            "{country}空军装备{num}架{aircraft}",
            "{country}研发{weapon}投入{amount}",
            "{country}与{country2}就{region}问题举行军事会谈",
            "{country}{branch}完成年度训练考核",
            "{country}首款隐身{aircraft}量产交付",
            "{weapon}技术突破：{country}实现{achievement}",
            "{country}维和部队在{region}执行任务",
            "{country}海军在{region}进行远洋训练",
        ],
        "country": ["中国", "美国", "俄罗斯", "日本", "韩国", "印度", "法国", "英国",
                    "德国", "以色列", "土耳其", "伊朗", "巴基斯坦", "澳大利亚"],
        "country2": ["中国", "美国", "俄罗斯", "日本", "韩国", "印度", "法国", "英国",
                     "德国", "以色列", "土耳其", "伊朗", "巴基斯坦", "澳大利亚"],
        "weapon": ["导弹防御系统", "高超音速导弹", "弹道导弹", "反舰导弹", "无人机",
                   "激光武器", "电磁炮", "反卫星武器", "防空系统", "巡航导弹",
                   "鱼雷", "核潜艇", "装甲车", "自行火炮", "察打一体无人机"],
        "aircraft": ["歼-20", "歼-35", "F-35", "F-22", "苏-57", "阵风", "台风",
                     "B-21", "运-20", "轰-20", "直-20", "全球鹰无人机", "翼龙无人机", "彩虹无人机"],
        "ship": ["福建舰", "山东舰", "辽宁舰", "福特号", "伊丽莎白女王号",
                 "055型驱逐舰", "052D型驱逐舰", "076型两栖攻击舰",
                 "阿利伯克级驱逐舰", "北风之神级核潜艇"],
        "branch": ["陆军", "海军", "空军", "火箭军", "战略支援部队", "海军陆战队", "特种部队"],
        "region": ["台海", "南海", "东海", "中东", "波罗的海", "印太", "黑海",
                   "朝鲜半岛", "地中海", "北极", "非洲", "中亚"],
        "conflict": ["俄乌冲突", "巴以冲突", "红海局势", "叙利亚局势", "也门局势"],
        "war_event": ["双方在前线激烈交火", "和谈取得进展", "平民伤亡人数上升",
                      "新一轮停火协议达成", "人道主义援助通道开放"],
        "mil_event": ["阅兵式", "国际防务展", "航展", "军事论坛", "武器博览会"],
        "org": ["北约", "联合国", "上合组织", "东盟", "欧盟", "AUKUS"],
        "leader": ["国防部长", "总统", "总参谋长", "国家领导人"],
        "sat_type": ["侦察", "通信", "导航", "预警", "遥感"],
        "waterway": ["台湾海峡", "马六甲海峡", "苏伊士运河", "霍尔木兹海峡", "对马海峡",
                     "巴士海峡", "宫古海峡"],
        "tonnage": ["5000", "10000", "20000", "40000", "70000", "100000"],
        "range": ["300", "500", "1000", "2000", "5000", "8000", "12000", "15000"],
        "num": ["2", "3", "5", "8", "10", "12", "20", "30", "50", "100", "200"],
        "gen": ["四", "五", "六"],
        "amount": ["500亿", "1000亿", "2000亿", "5000亿", "8000亿美元", "1万亿"],
        "year": ["2024", "2025", "2026"],
        "achievement": ["全天候作战能力", "超远程精确打击", "自主目标识别",
                        "多弹头分导技术", "隐身突防能力", "反隐身探测"],
    },
}


# ============================================================
# 2. 模板填充函数
# ============================================================

def fill_template(template: str, variables: dict) -> str:
    """随机填充模板中的占位符，同一模板中同名占位符每次独立随机"""
    result = template
    max_iter = 30
    for _ in range(max_iter):
        start = result.find("{")
        if start == -1:
            break
        end = result.find("}", start)
        if end == -1:
            break
        key = result[start + 1:end]
        if key in variables:
            value = random.choice(variables[key])
            result = result[:start] + value + result[end + 1:]
        else:
            # 未知占位符，跳过避免死循环
            result = result[:start] + result[start + 1:]
    return result


# ============================================================
# 3. 生成数据
# ============================================================

def generate_data(total_count=10000):
    """生成指定数量的新闻分类数据，各类别均匀分布"""
    intents = list(INTENT_TEMPLATES.keys())
    per_intent = total_count // len(intents)
    remainder = total_count - per_intent * len(intents)

    all_samples = []

    for idx, intent in enumerate(intents):
        config = INTENT_TEMPLATES[intent]
        templates = config["templates"]
        variables = {k: v for k, v in config.items() if k != "templates"}

        target_count = per_intent + (1 if idx < remainder else 0)
        generated = set()
        attempts = 0

        while len(generated) < target_count and attempts < target_count * 20:
            template = random.choice(templates)
            query = fill_template(template, variables)
            if query not in generated:
                generated.add(query)
            attempts += 1

        for query in generated:
            all_samples.append({"query": query, "intent": intent})

        actual = len(generated)
        print(f"  ✅ {intent:>15}: {actual} 条 (目标 {target_count})")

    random.shuffle(all_samples)
    return all_samples


# ============================================================
# 4. 保存为 CSV
# ============================================================

def save_csv(samples, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "intent"])
        writer.writeheader()
        writer.writerows(samples)


# ============================================================
# 5. 主函数
# ============================================================

def main():
    print("=" * 55)
    print("📰 新闻分类数据生成器")
    print("=" * 55)
    print(f"类别: {list(INTENT_TEMPLATES.keys())}")
    print(f"目标: 10,000 条\n")

    print("🔄 正在生成...")
    samples = generate_data(total_count=10000)

    print(f"\n📊 最终统计:")
    print(f"  总量: {len(samples)} 条")
    counter = Counter(s["intent"] for s in samples)
    for intent, cnt in sorted(counter.items()):
        print(f"  {intent:>15}: {cnt} 条 ({cnt / len(samples) * 100:.1f}%)")

    filepath = "data/raw/news_intent_data.csv"
    save_csv(samples, filepath)
    print(f"\n💾 已保存: {filepath}")

    print(f"\n📋 样例 (前 10 条):")
    print("-" * 80)
    for s in samples[:10]:
        print(f"  [{s['intent']:>15}] {s['query']}")
    print("-" * 80)
    print("\n✅ 第一步完成！接下来运行 data/data_process.py 进行数据划分")


if __name__ == "__main__":
    main()