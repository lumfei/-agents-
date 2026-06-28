"""
多 Agent 客服分流系统 - Mock 数据生成器
运行: python seed_mock_data.py
生成所有模拟数据到 数据/ 目录下
"""

import json
import random
import os
from datetime import datetime, timedelta

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(42)  # 固定种子，保证每次生成一致

# ============================================================
# 基础数据
# ============================================================

PRODUCTS_TEMPLATE = [
    # 数码产品
    {"product_id": "P001", "name": "iPhone 16 Pro Max 256G", "category": "手机", "price": 9999, "brand": "Apple", "warranty_months": 12},
    {"product_id": "P002", "name": "iPhone 16 128G", "category": "手机", "price": 5999, "brand": "Apple", "warranty_months": 12},
    {"product_id": "P003", "name": "MacBook Air M4 16G/512G", "category": "笔记本", "price": 10999, "brand": "Apple", "warranty_months": 12},
    {"product_id": "P004", "name": "MacBook Pro 14 M4 Pro 24G/1T", "category": "笔记本", "price": 19999, "brand": "Apple", "warranty_months": 12},
    {"product_id": "P005", "name": "iPad Pro M4 13寸 256G", "category": "平板", "price": 8499, "brand": "Apple", "warranty_months": 12},
    {"product_id": "P006", "name": "AirPods Pro 3", "category": "配件", "price": 1999, "brand": "Apple", "warranty_months": 6},
    {"product_id": "P007", "name": "Apple Watch Ultra 3", "category": "穿戴", "price": 5999, "brand": "Apple", "warranty_months": 12},
    {"product_id": "P008", "name": "Samsung Galaxy S25 Ultra", "category": "手机", "price": 8999, "brand": "Samsung", "warranty_months": 12},
    {"product_id": "P009", "name": "华为 Mate 70 Pro", "category": "手机", "price": 7999, "brand": "华为", "warranty_months": 12},
    {"product_id": "P010", "name": "小米 15 Pro", "category": "手机", "price": 4999, "brand": "小米", "warranty_months": 12},
    {"product_id": "P011", "name": "ThinkPad X1 Carbon Gen 12", "category": "笔记本", "price": 12999, "brand": "Lenovo", "warranty_months": 24},
    {"product_id": "P012", "name": "Dell XPS 16", "category": "笔记本", "price": 15999, "brand": "Dell", "warranty_months": 12},
    {"product_id": "P013", "name": "罗技 MX Master 3S 鼠标", "category": "配件", "price": 799, "brand": "罗技", "warranty_months": 12},
    {"product_id": "P014", "name": "索尼 WH-1000XM6 头戴式耳机", "category": "配件", "price": 2999, "brand": "索尼", "warranty_months": 12},
    {"product_id": "P015", "name": "Anker 100W 氮化镓充电器", "category": "配件", "price": 299, "brand": "Anker", "warranty_months": 18},
    # 生活用品
    {"product_id": "P016", "name": "戴森 V18 无线吸尘器", "category": "家电", "price": 4999, "brand": "戴森", "warranty_months": 24},
    {"product_id": "P017", "name": "小米空气净化器 5 Pro", "category": "家电", "price": 2999, "brand": "小米", "warranty_months": 12},
    {"product_id": "P018", "name": "飞利浦电动牙刷 HX9999", "category": "个护", "price": 1299, "brand": "飞利浦", "warranty_months": 24},
    {"product_id": "P019", "name": "极米 H6 投影仪 4K", "category": "家电", "price": 6999, "brand": "极米", "warranty_months": 12},
    {"product_id": "P020", "name": "Breville 878 咖啡机", "category": "家电", "price": 5499, "brand": "Breville", "warranty_months": 24},
    {"product_id": "P021", "name": "Nespresso Vertuo Next 胶囊咖啡机", "category": "家电", "price": 1499, "brand": "Nespresso", "warranty_months": 24},
    {"product_id": "P022", "name": "Yeelight Pro 智能灯带 5m", "category": "智能家居", "price": 399, "brand": "Yeelight", "warranty_months": 12},
    # 服装鞋帽
    {"product_id": "P023", "name": "Arc'teryx Beta AR 冲锋衣", "category": "户外", "price": 5999, "brand": "Arc'teryx", "warranty_months": 0},
    {"product_id": "P024", "name": "Nike Air Max 2026 跑鞋", "category": "运动", "price": 1499, "brand": "Nike", "warranty_months": 3},
    {"product_id": "P025", "name": "Lululemon Define 女士夹克", "category": "运动", "price": 1280, "brand": "Lululemon", "warranty_months": 0},
]

CUSTOMER_NAMES = [
    "张伟", "王芳", "李强", "赵敏", "刘洋", "陈静", "杨磊", "黄丽", "周杰", "吴婷",
    "徐明", "孙莉", "马超", "朱红", "胡波", "郭雪", "林峰", "何琳", "高远", "罗冰",
    "梁浩", "宋雨", "唐飞", "韩梅", "曹锐", "邓鑫", "彭海", "蒋雯", "曾辉", "沈丹",
]

CITY_NAMES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "重庆", "西安",
    "苏州", "天津", "长沙", "郑州", "东莞", "青岛", "沈阳", "宁波", "昆明", "大连",
]

STATUS_WEIGHTS_ORDER = {
    "pending_payment": 0.05,
    "paid": 0.05,
    "processing": 0.05,
    "shipped": 0.10,
    "in_transit": 0.15,
    "out_for_delivery": 0.05,
    "delivered": 0.40,
    "return_requested": 0.03,
    "returning": 0.02,
    "refunded": 0.05,
    "cancelled": 0.05,
}

PROBLEM_DESCRIPTIONS = [
    "商品收到后有外观瑕疵，屏幕有一道划痕",
    "电池续航明显低于宣传，正常使用不到半天",
    "无法正常开机，按电源键无反应",
    "收到后发现颜色与下单不一致，我要的是蓝色",
    "连接WiFi经常断连，信号不稳定",
    "系统升级后频繁闪退，影响正常使用",
    "充电接口松动，充电断断续续",
    "摄像头有进灰，拍照有明显黑点",
    "键盘有几个键按压不灵敏",
    "发货速度太慢，一周了还没有发出",
    "物流信息一周未更新，疑似丢件",
    "收到时外包装有明显压痕，怀疑运输损坏",
    "商品与描述不符，参数虚标",
    "配件缺少充电器和数据线",
    "蓝牙连接经常断开，距离稍远就没信号",
    "屏幕有亮点/坏点，在深色背景下明显",
    "指纹解锁成功率很低，十次只有两三次成功",
    "散热风扇噪音过大，正常使用时也呼呼响",
    "系统预装软件太多，很多无法卸载",
    "激活后发现序列号已被注册，怀疑是翻新机",
]

SOLUTION_NOTES = [
    "已安排上门取件退换，预计3个工作日内处理完毕",
    "已指导客户进行系统重置，问题已解决",
    "已补发配件，快递单号SF1234567890",
    "已退款至原支付账户，预计1-3个工作日到账",
    "已安排技术人员远程诊断，确认需返厂维修",
    "已提供优惠券补偿，客户接受方案",
    "已升级固件，问题已修复",
    "已协调仓库重新发货，加急处理",
    "确认属于运输损坏，已发起理赔流程",
    "已解释商品特性，客户表示理解并撤销投诉",
]

STATUS_WEIGHTS_REFUND = {
    "pending_approval": 0.15,
    "approved": 0.20,
    "rejected": 0.10,
    "refunding": 0.15,
    "completed": 0.35,
    "cancelled": 0.05,
}

CATEGORY_NAMES = ["手机", "笔记本", "平板", "配件", "穿戴", "家电", "个护", "户外", "运动", "智能家居"]

KNOWLEDGE_BASE_ARTICLES = [
    # 退货退款政策
    {"kb_id": "KB001", "title": "退货退款政策总则", "category": "政策", "tags": ["退货", "退款", "政策"],
     "content": """## 退货退款政策

1. **7天无理由退货**：自签收之日起7日内，商品完好、不影响二次销售，可申请无理由退货。
2. **15天质量问题退换**：自签收之日起15日内，因商品质量问题，可申请退换货。
3. **一年质保**：自签收之日起一年内，非人为损坏的故障，提供免费维修服务。
4. **退回要求**：退货时请保留所有配件、赠品、包装盒及发票。
5. **退款时效**：审核通过后，退款将在1-3个工作日内原路返回。

**不予退货的情形**：
- 已拆封的影音制品、软件等
- 定制类商品（刻字、特殊尺寸等）
- 因个人原因造成的损坏
- 超过7天无理由退货期"""},

    {"kb_id": "KB002", "title": "退款金额计算规则", "category": "政策", "tags": ["退款", "金额", "计算"],
     "content": """## 退款金额计算规则

1. **全额退款**：商品质量问题或发错货，全额退款包括商品金额+运费。
2. **部分退款**：7天无理由退货，退还商品金额，运费由买家承担。
3. **折旧扣除**：已使用超过30天的商品退货，按日折旧率0.1%扣除。
4. **优惠券分摊**：使用优惠券的订单，退款时按比例分摊扣除。
5. **赠品处理**：赠品未退回的，按赠品价值在退款中扣除。"""},

    {"kb_id": "KB003", "title": "物流配送常见问题", "category": "物流", "tags": ["物流", "配送", "快递"],
     "content": """## 物流配送常见问题

### 配送时效
- 标准配送：下单后3-5个工作日
- 加急配送：下单后1-2个工作日（需额外收费）
- 偏远地区：下单后5-7个工作日

### 常见问题处理
1. **物流信息不更新**：超过48小时无更新，请联系客服核查
2. **包裹显示已签收但未收到**：先询问家人/同事/物业，若确认未收到，24小时内联系客服
3. **需要修改地址**：未发出前可联系客服修改；已发出需联系快递公司转寄
4. **拒收处理**：拒收后包裹退回仓库，确认后安排退款（扣除运费）"""},

    {"kb_id": "KB004", "title": "产品保修政策说明", "category": "政策", "tags": ["保修", "维修", "质保"],
     "content": """## 产品保修政策

### 保修期限
- 手机/平板/笔记本：12个月
- 配件类（耳机、充电器等）：6个月
- 大家电类：24个月
- 部分品牌有延保服务，以购买时约定为准

### 保修范围
- 非人为损坏的硬件故障
- 出厂质量问题
- 保修期内免费维修

### 不保修范围
- 人为损坏（摔落、进水、私自拆修）
- 电池自然损耗（容量低于80%除外）
- 外观件（划痕、磨损）
- 第三方配件或软件引起的问题"""},

    {"kb_id": "KB005", "title": "账户与订单管理", "category": "账户", "tags": ["账户", "订单", "管理"],
     "content": """## 账户与订单管理

### 修改订单信息
- 未支付订单：可在订单详情页自行取消
- 已支付未发货：联系客服修改地址或取消
- 已发货订单：需联系快递公司转寄或拒收

### 发票开具
- 电子发票：下单后可在订单详情页下载，一般1-3个工作日开出
- 纸质发票：随商品一起寄出，如需单独寄送请联系客服
- 增值税专用发票：需提供完整的开票资料，审核后开出

### 优惠券使用
- 每个订单限用一张优惠券
- 优惠券不可叠加使用
- 部分商品不参与优惠券活动"""},

    {"kb_id": "KB006", "title": "账号安全问题处理指南", "category": "安全", "tags": ["账号", "安全", "被盗", "密码"],
     "content": """## 账号安全问题处理

1. **账号被盗**：立即联系我们冻结账号，提供身份验证信息找回
2. **异常登录提醒**：如收到未授权的登录提醒，建议立即修改密码
3. **密码找回**：通过绑定手机号或邮箱重置密码
4. **账号注销**：需完成所有进行中的订单/退款后，联系客服申请注销
5. **隐私保护**：我们不会以任何理由索要您的密码或验证码"""},

    {"kb_id": "KB007", "title": "商品使用常见问题 - 手机类", "category": "使用指南", "tags": ["手机", "使用", "设置"],
     "content": """## 手机类商品使用指南

### 新手机首次使用
1. 长按电源键开机（首次可能需要充电5分钟）
2. 按照屏幕提示完成初始设置
3. 建议连接WiFi进行系统更新
4. 可通过iCloud/华为云/小米云恢复旧手机数据

### 常见问题
- **耗电快**：关闭后台刷新、降低屏幕亮度、关闭不必要的定位服务
- **发热严重**：避免边充电边使用大型应用，取下保护壳散热
- **存储空间不足**：清理缓存、卸载不常用应用、备份照片到云盘
- **系统卡顿**：重启设备、更新系统、清理后台应用"""},

    {"kb_id": "KB008", "title": "商品使用常见问题 - 笔记本类", "category": "使用指南", "tags": ["笔记本", "电脑", "使用", "设置"],
     "content": """## 笔记本类商品使用指南

### 新笔记本首次设置
1. 连接电源适配器再进行首次开机
2. macOS用户：按照设置助理完成配置
3. Windows用户：跳过可选联网步骤以加快初始设置
4. 建议开启自动更新

### 常见问题
- **电池续航低于预期**：开启省电模式、降低屏幕刷新率、关闭后台应用
- **运行噪音大**：检查风扇是否有异物、清理通风口灰尘、更新系统
- **无法连接WiFi**：重启路由器、忘记网络后重新连接、更新网卡驱动
- **系统运行缓慢**：磁盘清理、关闭开机自启项、增加内存（如支持）"""},

    {"kb_id": "KB009", "title": "支付问题常见问答", "category": "支付", "tags": ["支付", "扣款", "失败"],
     "content": """## 支付问题处理

### 支付失败原因
1. 银行卡余额不足
2. 银行支付限额已超
3. 网络连接不稳定
4. 银行系统维护
5. 信用卡CVV码错误

### 已扣款但订单未生成
- 通常为银行预授权冻结，24小时内自动释放
- 如超过24小时仍未到账，请提供扣款截图联系客服
- 无需重复下单，等待系统自动处理即可

### 分期付款
- 支持3/6/12/24期分期
- 具体费率以支付页面为准
- 分期手续费由银行收取"""},

    {"kb_id": "KB010", "title": "客户投诉处理流程", "category": "内部", "tags": ["投诉", "处理", "流程"],
     "content": """## 客户投诉处理流程

### 分级处理
1. **一级投诉**（普通问题）：客服直接处理，24小时内解决
2. **二级投诉**（涉及退款、赔偿）：需主管审核，48小时内出方案
3. **三级投诉**（高金额、法律风险）：升级至法务及管理层处理

### 补偿标准参考
- 发货延迟：≤3天补偿50元券，3-7天补偿100元券，>7天补偿200元券
- 商品瑕疵：按商品价值5%-20%补偿
- 客服态度问题：补偿100元券+道歉
- 物流丢件：全额退款+额外补偿订单金额20%（最高1000元）

### 重要原则
- 先安抚情绪，再处理问题
- 不轻易承诺做不到的事情
- 每一个投诉都是改进机会"""},

    {"kb_id": "KB011", "title": "发票开具说明及税率", "category": "财务", "tags": ["发票", "税率", "开票"],
     "content": """## 发票开具说明

### 发票类型
1. **电子普通发票**：默认开具，订单完成后1-3个工作日发至邮箱
2. **纸质普通发票**：随货寄出或单独邮寄（满99元免邮费）
3. **增值税专用发票**：需审核一般纳税人资质，审核通过后3-5个工作日开出

### 税率
- 一般商品：13%
- 图书/农产品：9%
- 服务类：6%
- 个人消费建议开具普通发票即可"""},

    {"kb_id": "KB012", "title": "会员等级与积分规则", "category": "账户", "tags": ["会员", "积分", "等级"],
     "content": """## 会员等级与积分规则

### 会员等级
| 等级 | 年消费门槛 | 权益 |
|------|-----------|------|
| 普通会员 | 注册即享 | 基础服务 |
| 银卡会员 | ≥2000元 | 生日礼包+专属客服 |
| 金卡会员 | ≥10000元 | 双倍积分+免运费+优先发货 |
| 钻石会员 | ≥50000元 | 三倍积分+专属管家+延保一年 |

### 积分规则
- 消费1元=1积分
- 积分有效期12个月
- 100积分=1元，可在下单时抵扣
- 积分兑换商品不享受会员权益"""},

    {"kb_id": "KB013", "title": "跨境商品购买须知", "category": "政策", "tags": ["跨境", "海关", "关税"],
     "content": """## 跨境商品购买须知

1. **清关时间**：跨境商品需经海关清关，通常需要3-7个工作日
2. **个人额度**：跨境电商个人年交易限额26000元
3. **税费说明**：跨境商品综合税率一般为9.1%-23.05%，订单中包含税费
4. **退货限制**：跨境商品不支持无理由退货，仅质量问题可退
5. **身份信息**：需提供真实姓名和身份证号用于报关
6. **物流查询**：可通过物流单号在海关总署网站查询清关进度"""},

    {"kb_id": "KB014", "title": "售后维修流程", "category": "售后", "tags": ["维修", "返厂", "售后"],
     "content": """## 售后维修流程

### 维修申请步骤
1. 联系客服描述问题，客服初步判断是否在保修范围
2. 客服创建维修工单，生成维修编号
3. 客户将商品寄回指定维修中心（或预约上门取件）
4. 维修中心检测并出具检测报告
5. 确认维修方案后开始维修（保修期内免费）
6. 维修完成后回寄给客户

### 维修时效
- 一般维修：收到商品后3-5个工作日
- 需更换配件：收到商品后5-7个工作日
- 重大故障：收到商品后7-15个工作日

### 备用机服务
- 金卡及以上会员可申请备用机
- 备用机需缴纳押金，维修完成归还后退还"""},

    {"kb_id": "KB015", "title": "价格保护政策", "category": "政策", "tags": ["价格", "保价", "差价"],
     "content": """## 价格保护政策

1. **价保期限**：自签收之日起7天内
2. **适用范围**：商品降价（不含秒杀、限量抢购等特殊活动）
3. **申请方式**：联系客服提供订单号，核实后退还差价
4. **差价计算**：以实际支付金额为基准，不含优惠券抵扣部分
5. **退款方式**：差价退还至原支付账户或账户余额

### 不适用情形
- 使用优惠券/满减活动的订单
- 秒杀、限时抢购等特殊活动
- 企业批量采购订单
- 二手/翻新商品"""},
]


# ============================================================
# 数据生成函数
# ============================================================

def generate_customers(count=50):
    customers = []
    for i, name in enumerate(CUSTOMER_NAMES):
        total_orders = random.randint(0, 50)
        total_spent = sum(random.randint(99, 9999) for _ in range(max(1, total_orders)))
        customers.append({
            "customer_id": f"CU{i+1:04d}",
            "name": name,
            "phone": f"1{random.choice(['38','39','86','88','50','58'])}{random.randint(10000000,99999999)}",
            "city": random.choice(CITY_NAMES),
            "level": random.choices(
                ["普通会员", "银卡会员", "金卡会员", "钻石会员"],
                weights=[0.5, 0.3, 0.15, 0.05]
            )[0],
            "total_orders": total_orders,
            "total_spent": total_spent,
            "points": total_spent,
            "register_date": (datetime.now() - timedelta(days=random.randint(30, 1095))).strftime("%Y-%m-%d"),
            "last_login": (datetime.now() - timedelta(days=random.randint(0, 60))).strftime("%Y-%m-%d %H:%M:%S"),
            "is_vip": random.random() < 0.15,
            "tags": random.sample(["高价值", "投诉倾向", "沉默用户", "新品爱好者", "企业客户"], k=random.randint(0, 2)),
        })
    return customers


def generate_orders(customers, products, count=200):
    orders = []
    for i in range(count):
        customer = random.choice(customers)
        product = random.choice(products)
        qty = random.choices([1, 1, 1, 2, 2, 3], weights=[0.5, 0.25, 0.1, 0.08, 0.05, 0.02])[0]
        amount = product["price"] * qty
        created = datetime.now() - timedelta(days=random.randint(0, 120), hours=random.randint(0, 23))
        status = random.choices(
            list(STATUS_WEIGHTS_ORDER.keys()),
            weights=list(STATUS_WEIGHTS_ORDER.values())
        )[0]

        shipping_company = random.choice(["顺丰速运", "京东快递", "圆通速递", "中通快递", "韵达快递", "EMS", "极兔速递"])
        tracking_no = f"{random.choice(['SF','JD','YT','ZT','YD','EM','JT'])}{random.randint(1000000000,9999999999)}"[:12]

        order = {
            "order_id": f"ORD{i+1:05d}",
            "customer_id": customer["customer_id"],
            "customer_name": customer["name"],
            "products": [{"product_id": product["product_id"], "name": product["name"], "qty": qty, "price": product["price"]}],
            "total_amount": amount,
            "actual_paid": round(amount * random.uniform(0.85, 1.0), 2) if amount > 500 else amount,
            "status": status,
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "shipping_address": f"{customer['city']}{random.choice(['朝阳区','海淀区','浦东新区','天河区','高新区'])}{random.choice(['中山路','科技路','人民路','建设路'])}{random.randint(1,200)}号",
            "shipping_company": shipping_company,
            "tracking_no": tracking_no if status not in ["pending_payment"] else "",
            "payment_method": random.choice(["微信支付", "支付宝", "银行卡", "花呗"]),
            "remarks": random.choice(["", "", "", "急件，请尽快发货", "周末不要送货", "放快递柜", ""]),
        }
        orders.append(order)
    return orders


def generate_refunds(orders, count=40):
    refunds = []
    refund_sources = [o for o in orders if o["status"] in ["return_requested", "returning", "refunded", "delivered"]]
    if not refund_sources:
        refund_sources = orders

    for i in range(min(count, len(refund_sources))):
        order = random.choice(refund_sources)
        amount = order["actual_paid"] * random.uniform(0.3, 1.0)
        status = random.choices(
            list(STATUS_WEIGHTS_REFUND.keys()),
            weights=list(STATUS_WEIGHTS_REFUND.values())
        )[0]

        # 构造>1000的场景用于HITL演示
        exceed_hitl = amount > 1000 and status in ["pending_approval", "approved"]

        refunds.append({
            "refund_id": f"RF{i+1:04d}",
            "order_id": order["order_id"],
            "customer_id": order["customer_id"],
            "customer_name": order["customer_name"],
            "amount": round(amount, 2),
            "reason": random.choice(PROBLEM_DESCRIPTIONS),
            "status": status,
            "type": random.choice(["退货退款", "仅退款", "差价补偿"]),
            "hitl_required": exceed_hitl,
            "hitl_approved": None if exceed_hitl else random.choice([True, False, None]),
            "created_at": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d %H:%M:%S"),
            "processed_at": (datetime.now() - timedelta(days=random.randint(0, 15))).strftime("%Y-%m-%d %H:%M:%S") if status in ["completed", "rejected"] else None,
            "solution_note": random.choice(SOLUTION_NOTES) if status in ["completed", "approved"] else "",
        })
    return refunds


def generate_logistics(orders, count=80):
    logistics = []
    shippable = [o for o in orders if o.get("tracking_no") and o["status"] not in ["pending_payment", "paid", "cancelled"]]
    if not shippable:
        shippable = orders[:count]

    statuses = ["已揽收", "运输中", "到达中转站", "派送中", "已签收", "异常"]
    status_weights = [0.05, 0.20, 0.10, 0.15, 0.45, 0.05]

    for i, order in enumerate(shippable[:count]):
        log_status = random.choices(statuses, weights=status_weights)[0]
        created = datetime.now() - timedelta(days=random.randint(1, 30))
        tracking_events = []

        # 生成物流轨迹
        station_names = ["广州分拣中心", "深圳分拣中心", "北京分拣中心", "上海分拣中心", "杭州中转站",
                         "武汉中转站", "成都中转站", "南京分拣中心", "重庆中转站", "西安分拣中心"]
        for j in range(random.randint(2, 6)):
            ts = created + timedelta(hours=j * random.randint(3, 12))
            if ts > datetime.now():
                break
            tracking_events.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "location": random.choice(station_names),
                "event": random.choice(["包裹到达", "包裹已发出", "正在分拣", "安检通过", "装车完毕"]),
                "operator": f"{random.choice(['张','李','王','刘','陈'])}{random.choice(['师傅','专员','组长'])}",
            })

        if log_status == "已签收":
            tracking_events.append({
                "timestamp": (created + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d %H:%M:%S"),
                "location": order.get("shipping_address", "客户地址")[:10],
                "event": "已签收",
                "operator": "快递员",
            })

        logistics.append({
            "tracking_no": order["tracking_no"] or f"SF{1000000000+i}",
            "order_id": order["order_id"],
            "company": order["shipping_company"],
            "current_status": log_status,
            "current_location": random.choice(station_names),
            "estimated_delivery": (created + timedelta(days=random.randint(2, 5))).strftime("%Y-%m-%d"),
            "tracking_events": tracking_events,
            "recipient_name": order["customer_name"],
            "recipient_address": order["shipping_address"],
        })
    return logistics


def generate_conversations(customers, count=30):
    conversations = []
    intents = [
        ("查询订单状态", "技术支持"),
        ("申请退款退货", "财务"),
        ("物流问题", "售后"),
        ("商品使用咨询", "技术支持"),
        ("账户问题", "技术支持"),
        ("投诉", "售后"),
        ("发票问题", "财务"),
        ("价格咨询", "销售"),
    ]
    resolutions = ["已解决", "已升级人工", "客户撤销", "处理中"]

    for i in range(count):
        customer = random.choice(customers)
        intent, expected_agent = random.choice(intents)
        created = datetime.now() - timedelta(days=random.randint(0, 60), hours=random.randint(0, 23))

        messages = []
        # 客户消息
        msgs = [
            f"你好，我想咨询一下{customer.get('order_id','我的订单')}的问题",
            "请问有人能帮我看一下吗？",
            random.choice(PROBLEM_DESCRIPTIONS),
        ]
        for j, msg in enumerate(msgs):
            messages.append({
                "role": "customer",
                "content": msg,
                "timestamp": (created + timedelta(minutes=j * 2)).strftime("%Y-%m-%d %H:%M:%S"),
            })

        # 客服回复
        for j, reply in enumerate([
            random.choice(["您好，很高兴为您服务！", "感谢您的耐心等待！"]),
            random.choice(["我来查一下您的订单信息...", "您的问题我已经记录了"]),
            random.choice(SOLUTION_NOTES),
        ]):
            messages.append({
                "role": "agent",
                "content": reply,
                "agent_type": expected_agent,
                "timestamp": (created + timedelta(minutes=j * 3 + 1)).strftime("%Y-%m-%d %H:%M:%S"),
            })

        conversations.append({
            "conversation_id": f"CONV{i+1:04d}",
            "customer_id": customer["customer_id"],
            "customer_name": customer["name"],
            "intent": intent,
            "expected_agent": expected_agent,
            "messages": messages,
            "resolution": random.choice(resolutions),
            "satisfaction_score": random.choices([1, 2, 3, 4, 5], weights=[0.02, 0.03, 0.10, 0.35, 0.50])[0],
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": random.randint(3, 45),
        })
    return conversations


# ============================================================
# 主生成逻辑
# ============================================================

def main():
    print("=" * 50)
    print("多 Agent 客服分流系统 - Mock 数据生成器")
    print("=" * 50)

    # 生成各类型数据
    print("\n[1/6] 生成产品数据...")
    products = PRODUCTS_TEMPLATE
    save_json(products, "products.json")
    print(f"       → {len(products)} 条产品")

    print("[2/6] 生成客户数据...")
    customers = generate_customers(30)
    save_json(customers, "customers.json")
    print(f"       → {len(customers)} 条客户")

    print("[3/6] 生成订单数据...")
    orders = generate_orders(customers, products, 200)
    save_json(orders, "orders.json")
    print(f"       → {len(orders)} 条订单")
    print(f"         状态分布:")
    for s in sorted(set(o["status"] for o in orders)):
        count = sum(1 for o in orders if o["status"] == s)
        print(f"           {s}: {count}")

    print("[4/6] 生成退款数据...")
    refunds = generate_refunds(orders, 40)
    save_json(refunds, "refunds.json")
    print(f"       → {len(refunds)} 条退款")
    hitl_count = sum(1 for r in refunds if r["hitl_required"])
    print(f"         其中需人工审批(HITL): {hitl_count} 条 (金额>1000)")

    print("[5/6] 生成物流数据...")
    logistics = generate_logistics(orders, 80)
    save_json(logistics, "logistics.json")
    print(f"       → {len(logistics)} 条物流记录")

    print("[6/6] 生成客服对话数据...")
    conversations = generate_conversations(customers, 30)
    save_json(conversations, "conversations.json")
    print(f"       → {len(conversations)} 条对话记录")

    # 知识库（静态数据）
    save_json(KNOWLEDGE_BASE_ARTICLES, "knowledge_base.json")
    print(f"\n[+] 知识库: {len(KNOWLEDGE_BASE_ARTICLES)} 篇文章")

    # 生成汇总报告
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_products": len(products),
        "total_customers": len(customers),
        "total_orders": len(orders),
        "total_refunds": len(refunds),
        "total_logistics": len(logistics),
        "total_conversations": len(conversations),
        "total_kb_articles": len(KNOWLEDGE_BASE_ARTICLES),
        "hitl_cases": hitl_count,
        "order_status_distribution": {s: sum(1 for o in orders if o["status"] == s) for s in sorted(set(o["status"] for o in orders))},
    }
    save_json(summary, "_summary.json")
    print(f"\n{'=' * 50}")
    print(f"✅ 全部完成! 共生成 {sum(summary[k] for k in summary if k.startswith('total_'))} 条数据")
    print(f"   数据存放于: {OUTPUT_DIR}")
    print(f"{'=' * 50}")


def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
