"""Step2 纯规则版 — 不加API, 查看匹配率"""
import pandas as pd
import re
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

def norm(c):
    c = str(c).strip()
    c = re.sub(r'^(SHSE|SZSE|SH|SZ)\.?', '', c)
    return c.zfill(6) if c.isdigit() else c

SW_LIST = [
    '煤炭','电力设备','电子','房地产','纺织服饰','非银金融',
    '钢铁','公用事业','国防军工','环保','机械设备','基础化工',
    '计算机','家用电器','建筑材料','建筑装饰','交通运输',
    '美容护理','农林牧渔','汽车','轻工制造','商贸零售',
    '社会服务','石油石化','食品饮料','通信','传媒',
    '医药生物','银行','有色金属','综合'
]

# 加载
strat = pd.read_csv('clean_strategies.csv')
acct = pd.read_csv('clean_accounts.csv')
our = set(norm(c) for c in pd.concat([strat['stock_code'], acct['stock_code']]))
our = {c for c in our if len(c)==6 and not c.startswith(('12','11','10','13','14','9'))}

# 获取名称
names = {}
try:
    import akshare as ak
    df = ak.stock_info_a_code_name()
    for _,r in df.iterrows():
        names[norm(str(r['code']))] = str(r['name'])
except:
    pass

for _,r in acct.iterrows():
    c = norm(r['stock_code'])
    n = str(r.get('stock_name',''))
    if n and n!='nan' and c not in names:
        names[c] = n

# 规则
rules = [
    (r'煤$|煤矿|煤业|焦煤|焦炭','煤炭'),
    (r'电力$|发电$|水电|火电|核电|风电|热电|绿电','公用事业'),
    (r'钢铁$|特钢|钢管','钢铁'),
    (r'矿业$|黄金|白银|铜业|铝业|铅锌|镍业|钴业|锂业|稀土|钨业','有色金属'),
    (r'石油|石化|油气|炼化','石油石化'),
    (r'化工$|化学|化肥|农药|塑料|橡胶|树脂|涂料','基础化工'),
    (r'建材|水泥|玻璃|陶瓷|石膏|防水','建筑材料'),
    (r'建筑|装饰|装修|幕墙|园林|钢结构|基建|工程|施工','建筑装饰'),
    (r'汽车|客车|轿车|汽配|轮胎|车桥','汽车'),
    (r'家电|空调|冰箱|洗衣机|彩电|厨电|小家电|照明','家用电器'),
    (r'纺织|服装|服饰|鞋业|家纺|毛纺|棉纺|化纤|丝绸|皮革','纺织服饰'),
    (r'轻工|造纸|包装|家具|家居|卫浴|玩具|文具|五金','轻工制造'),
    (r'制药|药业|医药|生物$|基因|细胞|免疫|疫苗|诊断|医疗器械|医疗设备|中药','医药生物'),
    (r'食品|饮料|白酒|啤酒|红酒|乳业|调味品|酱油|醋|食用油|休闲|烘焙|糖果|预制菜|肉制品','食品饮料'),
    (r'农业$|牧业|渔业|养殖|饲料|种子|种业|粮食','农林牧渔'),
    (r'房地产|地产|房产|置业|物业|园区','房地产'),
    (r'银行$','银行'),
    (r'证券|券商|保险|信托|期货','非银金融'),
    (r'交通|运输|物流|快递|航运|港口|高速|铁路|机场|航空|海运|远洋','交通运输'),
    (r'电子$|半导体|芯片|集成电路|晶圆|封装|PCB|LED|OLED|光学|光电子|激光|传感器|电容|电阻|电感|连接器','电子'),
    (r'通信|电信|5G|6G|基站|天线|射频|光通信|光纤|光缆|光模块|交换机|路由器','通信'),
    (r'计算机$|软件|云计算|大数据|AI|区块链|信创|信息安全|网络安全','计算机'),
    (r'传媒|游戏|影视|动漫|出版|广电|数字媒体|元宇宙|广告|营销|直播','传媒'),
    (r'军工|国防|航天|航空|舰船|兵器|导弹|卫星|火箭|雷达','国防军工'),
    (r'机械|设备|机床|泵|阀|轴承|齿轮|减速器|电机|风机|压缩机|液压|气动','机械设备'),
    (r'电力设备|电气|变压器|开关|断路器|继电器|配电|输电|变电|电缆|电线|光伏|储能|电池|锂电池|逆变器|太阳能','电力设备'),
    (r'商贸|零售|百货|超市|便利店|免税|电商|贸易|外贸|进出口|批发','商贸零售'),
    (r'旅游|酒店|餐饮|景区|旅行社|教育|培训|体育|检测|认证|会展|环保|环卫|污水|垃圾','社会服务'),
    (r'美容|护理|化妆|护肤|医美|整形|口腔|牙科','美容护理'),
]

mapping = {}
for c in our:
    n = names.get(c,'')
    for pat, ind in rules:
        if re.search(pat, n):
            mapping[c] = ind
            break

# 策略名补充
strategy_ind = {
    '煤炭':'煤炭','半导体':'电子','医药':'医药生物','医疗':'医药生物',
    '军工':'国防军工','计算机':'计算机','化工':'基础化工','食品':'食品饮料',
    '酒':'食品饮料','游戏':'传媒','通信':'通信','旅游':'社会服务','房地产':'房地产',
    '养殖':'农林牧渔','机器人':'机械设备','创业板':'综合','科创':'电子',
    '双创':'综合','红利':'综合','成长':'综合','形态':'综合',
    '动量':'综合','综合':'综合','均衡':'综合','杠铃':'综合',
    '300增强':'综合','500增强':'综合','800增强':'综合','1000增强':'综合','2000增强':'综合',
}
strategy_stocks = strat[['stock_code','strategy_name']].drop_duplicates()
for _,r in strategy_stocks.iterrows():
    c = norm(r['stock_code'])
    if c in mapping: continue
    for k,ind in strategy_ind.items():
        if k in str(r['strategy_name']):
            mapping[c] = ind
            break

# ETF
for c in our:
    if c in mapping: continue
    if c.startswith(('15','16','51','56','58')):
        mapping[c] = '综合'

unmapped = [c for c in our if c not in mapping]

with open('_step2_noapi_result.txt','w',encoding='utf-8') as f:
    f.write(f'total_stocks: {len(our)}\n')
    total_in_sw = sum(1 for v in mapping.values() if v in SW_LIST and v != '综合')
    total_zonghe = sum(1 for v in mapping.values() if v == '综合')
    f.write(f'mapped_by_rule: {total_in_sw}\n')
    f.write(f'mapped_as_zonghe: {total_zonghe}\n')
    f.write(f'unmapped: {len(unmapped)}\n\n')
    f.write(f'Industry Distribution:\n')
    ind_cnt = defaultdict(int)
    for c,ind in mapping.items():
        ind_cnt[ind] += 1
    for ind in sorted(ind_cnt, key=lambda x:-ind_cnt[x]):
        f.write(f'  {ind}: {ind_cnt[ind]}\n')
    f.write(f'\nUnmapped Stocks ({len(unmapped)}):\n')
    for c in unmapped[:100]:
        f.write(f'{c} {names.get(c,"")}\n')

print(f'Total stocks: {len(our)}')
print(f'Rule matched (specific industry): {total_in_sw}')
print(f'Rule matched (as zonghe): {total_zonghe}')
print(f'Unmapped: {len(unmapped)}')
print(f'Done. See _step2_noapi_result.txt')
