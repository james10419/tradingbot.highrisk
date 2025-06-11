import asyncio
import pandas as pd
import numpy as np
import datetime
import warnings
import os

# --- rpy2 설치 및 R 환경 설정 확인 ---
# 이 코드를 실행하기 전, 터미널에서 'pip install rpy2'를 실행해야 합니다.
# 또한 시스템에 R이 설치되어 있어야 합니다.
try:
    from rpy2.robjects import pandas2ri, r, RRuntimeError
    from rpy2.robjects.packages import importr
    R_IS_AVAILABLE = True
except ImportError:
    print("Warning: 'rpy2' is not installed. R analysis will be mocked. To enable R, run: pip install rpy2")
    R_IS_AVAILABLE = False

warnings.filterwarnings('ignore')

class RAnalyzer:
    """R 스크립트와의 연동을 관리하는 클래스"""
    def __init__(self, script_path='r_analysis_script_enhanced.R'):
        self.script_path = script_path
        self.is_ready = False
        if R_IS_AVAILABLE:
            self._setup_r_environment()

    def _setup_r_environment(self):
        """R 환경을 설정하고 분석 스크립트를 로드합니다."""
        if not os.path.exists(self.script_path):
            print(f"Error: R script '{self.script_path}' not found.")
            print("Please make sure the R script file is in the same directory.")
            return

        try:
            pandas2ri.activate()
            # R 스크립트 파일을 로드합니다.
            r.source(self.script_path)
            print("✅ R environment and analysis script loaded successfully.")
            self.is_ready = True
        except RRuntimeError as e:
            print(f"Error setting up R environment: {e}")
            print("Please ensure R is installed and the 'TTR' package is available (install.packages('TTR') in R).")
        except Exception as e:
            print(f"An unexpected error occurred during R setup: {e}")

    def analyze(self, df: pd.DataFrame) -> str:
        """Python DataFrame을 R로 보내 분석하고 신호를 받습니다."""
        if not self.is_ready:
            return self._mock_analysis(df) # R을 사용할 수 없을 때 가상 분석 실행

        try:
            # pandas DataFrame을 R의 DataFrame으로 변환
            r_df = pandas2ri.py2rpy(df)
            # R 환경에 로드된 'get_trading_signal' 함수를 호출
            signal_vector = r['get_trading_signal'](r_df)
            # R 벡터를 Python 문자열로 변환하여 반환
            return signal_vector[0]
        except Exception as e:
            print(f"\nError during R analysis: {e}")
            return "HOLD" # 에러 발생 시 안전하게 'HOLD' 반환

    def _mock_analysis(self, df: pd.DataFrame) -> str:
        """R을 사용할 수 없을 때 실행되는 가상 분석 함수"""
        if len(df) < 20: return "HOLD"
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        last_row = df.iloc[-1]
        volume_spike = last_row['volume'] > (last_row['volume_ma20'] * 3)
        price_increase = last_row['close'] > df.iloc[-2]['close']
        if volume_spike and price_increase: return "BUY"
        return "HOLD"

class ScalpingBot:
    """스캘핑 전략을 실행하는 메인 클래스"""
    def __init__(self, r_analyzer: RAnalyzer):
        self.r_analyzer = r_analyzer
        self.ohlcv_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.position = None  # 현재 포지션 상태 ('long', None)
        self.entry_price = 0
        self.trade_count = 0
        self.balance = 1000  # 가상 잔고 (USD)
        self.initial_balance = self.balance
        self.stop_loss_pct = 0.005  # 0.5% 손절
        self.take_profit_pct = 0.01  # 1% 익절

    async def _fetch_data_mock(self):
        """가상 실시간 데이터를 생성하는 제너레이터"""
        base_price = 70000
        trend = 0
        for i in range(1000): # 1000번의 틱 데이터 생성 후 종료
            # 트렌드 시뮬레이션
            if np.random.rand() < 0.01: trend = np.random.randn() * 0.5
            price_change = trend + np.random.randn() * 20

            # 거래량 급증 시뮬레이션 (3% 확률)
            volume = np.random.randint(1, 10)
            if np.random.rand() < 0.03:
                volume *= np.random.randint(10, 20)
                price_change *= 2.5
                print(f"\n--- ❗ Volume Spike Event ({volume} units) ---")

            base_price += price_change
            if base_price < 0: base_price = 10 # 가격이 음수가 되지 않도록 방지

            new_data = {
                'timestamp': datetime.datetime.now().timestamp() * 1000,
                'open': base_price,
                'high': base_price + np.random.rand() * 10,
                'low': base_price - np.random.rand() * 10,
                'close': base_price,
                'volume': volume
            }
            yield new_data
            await asyncio.sleep(0.5) # 0.5초마다 새로운 데이터 생성

    def _log_trade(self, action, price, profit_pct=None):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if profit_pct is not None:
            profit_usd = self.balance * profit_pct if action != "START" else 0
            self.balance += profit_usd
            print(f"✅ [{timestamp}] {action:^18} | Price: {price:,.2f} | P/L: {profit_pct:+.2%} ({profit_usd:+.2f} USD) | Balance: {self.balance:,.2f} USD")
        else:
            print(f"🔥 [{timestamp}] {action:^18} | Price: {price:,.2f} | Position Opened | Balance: {self.balance:,.2f} USD")

    async def run(self):
        print("🚀 Scalping Bot Started...")
        print(f"Initial Balance: {self.balance:,.2f} USD")
        print("Using R for Analysis:", R_IS_AVAILABLE and self.r_analyzer.is_ready)
        print("-" * 60)

        async for data in self._fetch_data_mock():
            new_row = pd.DataFrame([data])
            self.ohlcv_data = pd.concat([self.ohlcv_data, new_row], ignore_index=True)
            self.ohlcv_data['timestamp'] = pd.to_datetime(self.ohlcv_data['timestamp'], unit='ms')

            # 메모리 관리를 위해 오래된 데이터는 삭제
            if len(self.ohlcv_data) > 200:
                self.ohlcv_data = self.ohlcv_data.iloc[-200:]

            current_price = data['close']
            print(f"\r[{datetime.datetime.now().strftime('%H:%M:%S')}] Watching... Price: {current_price:,.2f}, Volume: {data['volume']}", end="")

            # 1. 위험 관리 (포지션 보유 시)
            if self.position == 'long':
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
                take_profit_price = self.entry_price * (1 + self.take_profit_pct)

                if current_price <= stop_loss_price:
                    self._log_trade("STOP-LOSS SELL", current_price, pnl_pct)
                    self.position = None
                    continue

                if current_price >= take_profit_price:
                    self._log_trade("TAKE-PROFIT SELL", current_price, pnl_pct)
                    self.position = None
                    continue

            # 2. 전략 분석 (포지션 미보유 시)
            if self.position is None and len(self.ohlcv_data) > 20:
                signal = self.r_analyzer.analyze(self.ohlcv_data.copy())
                
                if signal == "BUY":
                    self.position = 'long'
                    self.entry_price = current_price
                    self.trade_count += 1
                    self._log_trade("BUY", self.entry_price)
                elif signal == "SELL" and self.position == 'long':
                    # R 스크립트가 능동적으로 포지션 종료 신호를 줄 수도 있습니다.
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                    self._log_trade("STRATEGY SELL", current_price, pnl_pct)
                    self.position = None
        
        # 시뮬레이션 종료 후 결과 요약
        print("\n" + "="*60)
        print("SIMULATION FINISHED")
        print(f"Total Trades: {self.trade_count}")
        print(f"Final Balance: {self.balance:,.2f} USD")
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        print(f"Total Return: {total_return:+.2%}")
        print("="*60)


async def main():
    # R 분석기 초기화
    analyzer = RAnalyzer()
    
    # 봇 인스턴스 생성 및 실행
    bot = ScalpingBot(r_analyzer=analyzer)
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped manually.")

