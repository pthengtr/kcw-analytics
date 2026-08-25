"""Configurable mappings for Lazada reconciliation.

Accountants may add aliases, status groups, Freename mappings, and
Wallet Type/Sub Type mappings here. Unknown values must never be
silently discarded — they appear on the Unknown_Mappings sheet.
"""

from __future__ import annotations

from decimal import Decimal

PROGRAM_VERSION = "1.0.0"
PROGRAM_NAME = "lazada-reconciliation"

DEFAULT_TOLERANCE = Decimal("0.01")
MONEY_QUANT = Decimal("0.01")
HEADER_SCAN_ROWS = 30

INSUFFICIENT_DATA = "ข้อมูลไม่เพียงพอ"

# ---------------------------------------------------------------------------
# Header aliases (matched after trim + casefold). Do not rely on column index.
# ---------------------------------------------------------------------------

HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "order_number": (
        "ordernumber",
        "order number",
        "order no",
        "order no.",
        "order_id",
        "orderid",
        "หมายเลขคำสั่งซื้อ",
        "เลขที่คำสั่งซื้อ",
        "เลขออเดอร์",
    ),
    "paid_price": (
        "paidprice",
        "paid price",
        "paid_price",
        "unit paid price",
    ),
    "order_status": (
        "status",
        "order status",
        "orderstatus",
        "สถานะ",
        "สถานะคำสั่งซื้อ",
    ),
    "create_time": (
        "createtime",
        "create time",
        "created at",
        "order create time",
        "วันที่สร้างคำสั่งซื้อ",
        "วันที่สร้าง",
    ),
    "amount": (
        "amount",
        "transaction amount",
        "จำนวนเงิน(รวมภาษี)",
        "จำนวนเงิน (รวมภาษี)",
        "จำนวนเงิน",
        "amount (incl. tax)",
        "amountincltax",
    ),
    "freename": (
        "freename",
        "free name",
        "free_name",
        "transaction name",
        "transaction type",
        "fee name",
        "ชื่อรายการธุรกรรม",
        "ชื่อรายการ",
        "ประเภทรายการ",
    ),
    "vat_amount": (
        "vat amount",
        "vat",
        "vatamount",
    ),
    "transfer_status": (
        "สถานะการโอนเงิน",
        "transfer status",
        "payout status",
        "wallet status",
    ),
    "statement_code": (
        "รหัสรอบบิล",
        "statement no",
        "statement number",
        "statement code",
        "short code",
    ),
    "statement_period": (
        "ระยะเวลาใบแจ้งยอด",
        "statement period",
        "period",
    ),
    "order_item_id": (
        "orderitemid",
        "order item id",
        "รหัสสินค้าในคำสั่งซื้อ",
        "item id",
    ),
    "wallet_type": (
        "type",
        "transaction type",
        "ประเภท",
    ),
    "wallet_sub_type": (
        "sub type",
        "subtype",
        "sub-type",
        "sub_type",
        "ประเภทย่อย",
    ),
    "remarks": (
        "remarks",
        "remark",
        "comment",
        "comments",
        "ความคิดเห็น",
        "หมายเหตุ",
    ),
    "transaction_time": (
        "transaction time",
        "transaction date",
        "time",
        "วันที่ทำรายการ",
        "วันที่ปรับปรุงเข้ายอดของฉัน",
    ),
    "transaction_number": (
        "transaction number",
        "transaction no",
        "txn number",
        "เลขที่รายการ",
    ),
}

# Distinctive columns used to identify file type. paid_price is orders-only.
FILE_TYPE_REQUIRED: dict[str, tuple[str, ...]] = {
    "orders": ("order_number", "paid_price", "order_status"),
    "finance_transaction": ("order_number", "amount", "freename"),
    "finance_overview": ("order_number", "amount", "transfer_status"),
    "wallet": ("wallet_type", "wallet_sub_type", "amount"),
}

FILE_TYPE_LABELS = {
    "orders": "คำสั่งซื้อทั้งหมด (Order list)",
    "finance_transaction": "รายการทางบัญชีแบบธุรกรรม (Freename / ชื่อรายการธุรกรรม)",
    "finance_overview": "รายการทางบัญชีแบบสรุปออเดอร์ (Income Order Overview)",
    "wallet": "ยอดของฉัน (Wallet / Balance Transactions)",
}

# ---------------------------------------------------------------------------
# Order status groups (case-insensitive exact match after trim)
# ---------------------------------------------------------------------------

STATUS_GROUPS: dict[str, str] = {
    "confirmed": "success",
    "delivered": "success",
    "shipped": "success",
    "packed": "success",
    "ready to ship": "success",
    "ready_to_ship": "success",
    "ยืนยันแล้ว": "success",
    "จัดส่งแล้ว": "success",
    "สำเร็จ": "success",
    "canceled": "cancelled",
    "cancelled": "cancelled",
    "cancel": "cancelled",
    "ยกเลิก": "cancelled",
    "refunded": "refund",
    "refund": "refund",
    "returned": "refund",
    "package returned": "refund",
    "in transit: returning to seller": "refund",
    "คืนเงิน": "refund",
    "คืนสินค้า": "refund",
    "pending": "other",
    "unpaid": "other",
}

# ---------------------------------------------------------------------------
# Finance Freename mapping (case-insensitive exact, then contains)
# Values: order_income, shipping_fee, service_fee, other_fee, refund,
#         adjustment, unknown
# ---------------------------------------------------------------------------

FREENAME_EXACT: dict[str, str] = {
    "shipping fee voucher refund to lazada": "shipping_fee",
    "shipping fee": "shipping_fee",
    "shipping fee (charged by lazada)": "shipping_fee",
    "shipping fee charged by lazada": "shipping_fee",
    "order item settlement": "order_income",
    "item price": "order_income",
    "item price credit": "order_income",
    "payment fee": "service_fee",
    "commission": "service_fee",
    "commission fee": "service_fee",
    "order item commission fee": "service_fee",
    "transaction fee": "service_fee",
    "service fee": "service_fee",
    "promotional charges": "other_fee",
    "promo credits": "adjustment",
    "refund": "refund",
    "refund to customer": "refund",
    "lost or damaged package compensation": "adjustment",
    "adjustment": "adjustment",
    "other": "other_fee",
}

FREENAME_CONTAINS: tuple[tuple[str, str], ...] = (
    ("shipping fee voucher refund to lazada", "shipping_fee"),
    ("shipping fee", "shipping_fee"),
    ("ค่าขนส่ง", "shipping_fee"),
    ("commission", "service_fee"),
    ("payment fee", "service_fee"),
    ("ค่าบริการ", "service_fee"),
    ("ค่าธรรมเนียม", "service_fee"),
    ("wht", "other_fee"),
    ("withholding", "other_fee"),
    ("refund", "refund"),
    ("คืนเงิน", "refund"),
    ("adjustment", "adjustment"),
    ("ปรับปรุง", "adjustment"),
)

FEE_CATEGORIES = (
    "order_income",
    "shipping_fee",
    "service_fee",
    "other_fee",
    "refund",
    "adjustment",
    "unknown",
)

# ---------------------------------------------------------------------------
# Wallet Type + Sub Type mapping (case-insensitive)
# Values: settlement_inflow, bank_withdrawal, opening_balance,
#         closing_balance, adjustment, unknown
# ---------------------------------------------------------------------------

WALLET_EXACT: dict[tuple[str, str], str] = {
    ("deposit", "settlement"): "settlement_inflow",
    ("withdrawal", "auto withdrawal"): "bank_withdrawal",
    ("withdrawal", "withdrawal"): "bank_withdrawal",
    ("deposit", "opening balance"): "opening_balance",
    ("deposit", "opening"): "opening_balance",
    ("balance", "opening balance"): "opening_balance",
    ("deposit", "closing balance"): "closing_balance",
    ("balance", "closing balance"): "closing_balance",
    ("adjustment", "adjustment"): "adjustment",
    ("deposit", "adjustment"): "adjustment",
    ("withdrawal", "adjustment"): "adjustment",
}

WALLET_TYPE_CONTAINS: tuple[tuple[str, str], ...] = (
    ("auto withdrawal", "bank_withdrawal"),
    ("opening", "opening_balance"),
    ("closing", "closing_balance"),
    ("settlement", "settlement_inflow"),
)

# ---------------------------------------------------------------------------
# Rows that must never be treated as orders / transactions
# ---------------------------------------------------------------------------

# Strong markers always drop the row. Weak markers ("total"/"รวม") drop the row
# only when they sit on an identifier or the order number is empty — never when
# they appear as a product name on a real order.
STRONG_TOTAL_ROW_MARKERS = (
    "grand total",
    "grandtotal",
    "รวมทั้งหมด",
    "ยอดรวม",
)
WEAK_TOTAL_ROW_MARKERS = (
    "total",
    "รวม",
)
TOTAL_ROW_MARKERS = STRONG_TOTAL_ROW_MARKERS + WEAK_TOTAL_ROW_MARKERS

# Customer / personal columns — never copied to output or logs.
PII_HEADER_HINTS = (
    "customername",
    "customeremail",
    "customerphone",
    "nationalregistration",
    "shippingname",
    "shippingaddress",
    "shippingphone",
    "shippingcity",
    "shippingpost",
    "shippingcountry",
    "shippingregion",
    "billingname",
    "billingaddr",
    "billingphone",
    "billingcity",
    "billingpost",
    "billingcountry",
    "taxcode",
    "sellernote",
    "buyerfaileddeliveryusername",
    "ชื่อลูกค้า",
    "ที่อยู่",
    "เบอร์",
    "โทรศัพท์",
    "อีเมล",
)

# Extra columns we keep only when needed for audit (never PII).
ORDERS_KEEP = (
    "order_number",
    "paid_price",
    "order_status",
    "create_time",
    "source_file",
    "source_row",
)
FINANCE_KEEP = (
    "order_number",
    "amount",
    "freename",
    "vat_amount",
    "transfer_status",
    "order_status",
    "statement_code",
    "statement_period",
    "order_item_id",
    "source_file",
    "source_row",
    "mapped_category",
    "finance_profile",
    "signed_amount",
    "absolute_amount",
)
WALLET_KEEP = (
    "wallet_type",
    "wallet_sub_type",
    "amount",
    "remarks",
    "transaction_time",
    "transaction_number",
    "source_file",
    "source_row",
    "mapped_category",
    "signed_amount",
    "absolute_amount",
)

MATCH_STATUSES = (
    "MATCHED",
    "NET_SETTLED",
    "ORDER_NOT_RELEASED",
    "FINANCE_FROM_OTHER_PERIOD",
    "AMOUNT_MISMATCH",
    "CANCELLED_OR_REFUNDED",
    "UNKNOWN",
)

POSSIBLE_REASONS = {
    "ORDER_NOT_RELEASED": "ลูกค้ายังไม่กดยืนยันรับสินค้า เงินจึงยังไม่เข้า Wallet",
    "FINANCE_FROM_OTHER_PERIOD": "เป็นรายได้จากออเดอร์เดือนก่อน หรืออยู่นอกไฟล์คำสั่งซื้อเดือนนี้",
    "CANCELLED_OR_REFUNDED": "มีการคืนเงินหรือยกเลิก",
    "AMOUNT_MISMATCH": "มียอดไม่ตรงกัน อาจมีค่าปรับหรือรายการปรับปรุง",
    "NET_SETTLED": (
        "ยอด Finance เป็นยอดสุทธิหลังหักค่าธรรมเนียม ไม่ใช่ paidPrice "
        "ผลต่างคือค่าธรรมเนียมโดยประมาณ (implied fee)"
    ),
    "MATCHED": "มียอดในทั้งสองไฟล์และผลต่างอยู่ใน tolerance",
    "EMPTY_ORDER_NUMBER": "ข้อมูล orderNumber ว่างหรือรูปแบบไม่ตรงกัน",
    "UNKNOWN": "วิเคราะห์ไม่ได้จากข้อมูลที่มี",
}
