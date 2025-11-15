import sys
import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from tinkoff.invest import Client, MoneyValue, Quotation, InstrumentIdType, OperationState
import traceback

# Константы
TOKEN_FILE = "token.txt"
OP_TYPE_INPUT = '1'
OP_TYPE_OUTPUT = '9'
OP_TYPE_BROKER_COMMISSION = '19'
OP_TYPE_EXCHANGE_COMMISSION = '20'
OP_TYPE_SERVICE_COMMISSION = '18'
OP_TYPE_MARGIN_COMMISSION = '17'
OP_TYPE_BUY = '10'
OP_TYPE_SELL = '11'
OP_TYPE_BUY_CARD = '13'
OP_TYPE_DIVIDEND = '21'
OP_TYPE_TAX_DIVIDEND = '8'
OP_TYPE_COUPON = '23'
OP_TYPE_TAX_COUPON = '2'
OP_TYPE_TAX = '5'
OP_TYPE_TAX_LUCRE = '6'
OP_TYPE_BENEFIT_TAX = '7'

# Преобразование данных
def money_value_to_float(money_value: MoneyValue | None) -> float:
    if not money_value:
        return 0.0
    return money_value.units + money_value.nano / 1e9

def quotation_to_float(quotation: Quotation | None) -> float:
    if not quotation:
        return 0.0
    return quotation.units + quotation.nano / 1e9

def format_number(value: float | int | None, precision: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{value:,.{precision}f}".replace(",", " ")
    except (ValueError, TypeError):
        return str(value)

def format_currency(value: float | int | None, currency: str = "₽", precision: int = 2) -> str:
    if value is None:
        return "- ₽"
    formatted_value = format_number(value, precision)
    if currency.lower() in ["rub", "₽"]:
        actual_currency = "₽"
    elif currency.lower() == "usd":
        actual_currency = "$"
    elif currency.lower() == "eur":
        actual_currency = "€"
    else:
        actual_currency = currency
    return f'{formatted_value} {actual_currency}'

# Кэш информации об инструментах
instrument_cache = {}
def get_instrument_info_cached(client: Client, figi: str) -> dict:
    if figi in instrument_cache:
        return instrument_cache[figi]

    try:
        response = client.instruments.get_instrument_by(id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi)
        instrument = response.instrument
        info = {
            'name': instrument.name,
            'ticker': instrument.ticker,
            'figi': instrument.figi,
            'currency': instrument.currency,
            'lot': instrument.lot,
            'class_code': instrument.class_code,
            'isin': instrument.isin,
            'type': instrument.instrument_type,
        }
        instrument_cache[figi] = info
        return info
    except Exception as e:
        print(f"Ошибка получения информации для FIGI {figi}: {e}")
        return {'name': f'Неизвестно ({figi})', 'ticker': '?', 'figi': figi, 'currency': '?', 'lot': 1, 'class_code': '?', 'isin': '?', 'type': '?'}

# Обработка данных API
def process_operations(operations_response) -> list:
    processed = []
    if not operations_response or not operations_response.operations:
        return processed

    for oper in operations_response.operations:
        processed.append({
            'id': oper.id,
            'parent_operation_id': oper.parent_operation_id,
            'currency': oper.currency,
            'payment_value': money_value_to_float(oper.payment),
            'price_value': money_value_to_float(oper.price),
            'state': oper.state,
            'quantity': oper.quantity,
            'quantity_rest': oper.quantity_rest,
            'figi': oper.figi,
            'instrument_type': oper.instrument_type,
            'date': oper.date,
            'type': oper.type,
            'operation_type': str(oper.operation_type),
            'trades': oper.trades,
        })
    return processed

def process_positions(client: Client, positions: list) -> list:
    processed_positions = []
    if not positions:
        return processed_positions

    for pos in positions:
        if not pos.figi or pos.quantity is None:
            continue

        instrument_info = get_instrument_info_cached(client, pos.figi)
        current_price_float = money_value_to_float(pos.current_price)
        avg_price_float = money_value_to_float(pos.average_position_price)
        quantity_float = quotation_to_float(pos.quantity)
        expected_yield_float = money_value_to_float(pos.expected_yield)

        cost_float = quantity_float * current_price_float

        processed_positions.append({
            'Имя': instrument_info['name'],
            'Тикер': instrument_info['ticker'],
            'figi': pos.figi,
            'Средняя цена_float': avg_price_float,
            'Текущая цена_float': current_price_float,
            'Валюта': instrument_info['currency'],
            'Количество_float': quantity_float,
            'Стоимость_float': cost_float,
            'Доход_float': expected_yield_float,
            'Лот': instrument_info['lot'],
            'Тип': instrument_info['type'],
        })
    return sorted(processed_positions, key=lambda x: x['Стоимость_float'], reverse=True)

def process_account_info(portfolio_response) -> dict:
    if not portfolio_response:
        return {}

    return {
        'total_float': money_value_to_float(portfolio_response.total_amount_portfolio),
        'stocks_price_float': money_value_to_float(portfolio_response.total_amount_shares),
        'bonds_price_float': money_value_to_float(portfolio_response.total_amount_bonds),
        'etfs_price_float': money_value_to_float(portfolio_response.total_amount_etf),
        'currencies_price_float': money_value_to_float(portfolio_response.total_amount_currencies),
        'futures_price_float': money_value_to_float(portfolio_response.total_amount_futures),
        'current_id': portfolio_response.account_id,
        'current_positions': portfolio_response.positions,
        'relative_profitability_percent': quotation_to_float(portfolio_response.expected_yield),
    }

def get_token() -> str:
    token = None
    # Попытка прочитать токен из файла
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
                token = f.read().strip()
            if token:
                return token
        except Exception as e:
            print(f"Ошибка чтения файла {TOKEN_FILE}: {e}")

    # Если токен не найден или пуст, выдаём инструкцию
    token_url = "https://www.tinkoff.ru/invest/settings/"
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Требуется токен API")
    msg_box.setIcon(QMessageBox.Icon.Information)
    msg_box.setTextFormat(Qt.TextFormat.RichText)
    msg_box.setText(
        f"Не удалось найти действительный токен API в файле <code>{TOKEN_FILE}</code>.<br><br>"
        f"Пожалуйста, введите ваш API токен Тинькофф Инвестиций в следующем окне.<br><br>"
        f"Вы можете получить токен по ссылке:<br>"
        f"<a href='{token_url}'>{token_url}</a>"
    )
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
    ret = msg_box.exec()

    if ret == QMessageBox.StandardButton.Cancel:
        QMessageBox.critical(None, "Отмена", "Ввод токена отменён. Программа будет закрыта.")
        sys.exit(1)

    token, ok = QInputDialog.getText(None, "Введите токен", "Введите ваш API токен Тинькофф Инвестиций:")

    if not ok or not token.strip():
        QMessageBox.critical(None, "Ошибка", "Токен не введён. Программа будет закрыта.")
        sys.exit(1)

    token = token.strip()

    try:
        with open(TOKEN_FILE, 'w', encoding='utf-8') as f:
            f.write(token)
    except Exception as e:
        print(f"Не удалось сохранить токен в файл {TOKEN_FILE}: {e}")
        QMessageBox.warning(None, "Предупреждение", f"Не удалось сохранить токен в файл {TOKEN_FILE}:\n{e}")

    return token

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    token_value = get_token()
    main_window = None

    try:
        # Инициализация клиента Tinkoff API
        with Client(token_value) as client_instance:
            accounts_response = client_instance.users.get_accounts()
            all_accounts_list = [
                {"name": account.name, "id": account.id}
                for account in accounts_response.accounts
            ]

            # Проверка наличия счетов
            if not all_accounts_list:
                 QMessageBox.critical(None, "Ошибка", "Не удалось получить список счетов. Проверьте токен и соединение с интернетом.")
                 sys.exit(1)

            # Создание и отображение главного окна
            main_window = PortfolioApp(all_accounts_list, client_instance)
            main_window.show()
            sys.exit(app.exec())

    except Exception as e:
         if main_window:
             main_window.close()
         error_details = traceback.format_exc()
         QMessageBox.critical(None, "Критическая ошибка", f"Не удалось инициализировать приложение:\n{e}\n\nDetails:\n{error_details}")
         sys.exit(1)