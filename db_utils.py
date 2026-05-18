# -*- coding: utf-8 -*-
"""db_utils.py — 数据库读写工具（供 predictor.py 和 quant.py 共用）"""
import pymysql

DB_CONFIG = dict(host="localhost", user="root", password="root",
                 database="pl5_predictor", charset="utf8mb4")


def db_connect():
    return pymysql.connect(**DB_CONFIG)


def load_history(count: int = 100):
    """从 draws 表读取最近 count 期数据。
    返回 newest-first 列表: [{"issue":..., "date":..., "nums":[...]}, ...]"""
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT issue, open_date, d1, d2, d3, d4, d5 "
                "FROM draws ORDER BY open_date DESC LIMIT %s",
                (count,)
            )
            rows = cur.fetchall()
        return [
            {
                "issue": row[0],
                "date": str(row[1]),
                "nums": [int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6])],
            }
            for row in rows
        ]
    finally:
        conn.close()
