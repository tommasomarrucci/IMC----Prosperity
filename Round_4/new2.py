from datamodel import Order, TradingState, OrderDepth, ConversionObservation
from typing import Dict, List
import jsonpickle
import numpy as np
import math

# Constants and parameters for MACARON strategy
class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

PARAMS = {
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 1.0,
        "make_min_edge": 0.3,
        "init_make_edge": 0.9,
        "min_edge": 0.2,
        "volume_avg_timestamp": 5,
        "volume_bar": 50,
        "dec_edge_discount": 0.8,
        "step_size": 0.2,
        "CSI": 40.0,
        "sun_bias_edge": 0.2,
        "conv_limit": 10,
        "sugar_momentum_window": 3,
        "sugar_bias_edge": 0.1
    }
}

# Combined Trader handles MACARONS, base commodities, and options
class Trader:
    def __init__(self):
        # MACARON strategy state
        self.params = PARAMS
        self.LIMIT = {Product.MAGNIFICENT_MACARONS: 75}
        # Base & options strategy state
        self.last_prices: Dict[str, float] = {}
        self.underlying_symbol = "VOLCANIC_ROCK"
        self.option_symbols = [
            ("VOLCANIC_ROCK_VOUCHER_9500", 9500),
            ("VOLCANIC_ROCK_VOUCHER_9750", 9750),
            ("VOLCANIC_ROCK_VOUCHER_10000", 10000),
            ("VOLCANIC_ROCK_VOUCHER_10250", 10250),
            ("VOLCANIC_ROCK_VOUCHER_10500", 10500)
        ]
        self.risk_free_rate = 0.0
        self.implied_volatility = 0.18

    # ----- Helper functions -----
    def implied_bid_ask(self, obs: ConversionObservation):
        bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
        ask = obs.askPrice + obs.importTariff + obs.transportFees
        return bid, ask

    def adaptive_edge(self, curr_edge: float, position: int, traderObject: dict) -> float:
        p = self.params[Product.MAGNIFICENT_MACARONS]
        hist = traderObject.setdefault("volume_history", [])
        hist.append(abs(position))
        if len(hist) > p["volume_avg_timestamp"]:
            hist.pop(0)
        if len(hist) < p["volume_avg_timestamp"]:
            return curr_edge
        avg_vol = np.mean(hist)
        if avg_vol >= p["volume_bar"]:
            traderObject["volume_history"] = []
            return curr_edge + p["step_size"]
        if p["dec_edge_discount"] * p["volume_bar"] * (curr_edge - p["step_size"]) > avg_vol * curr_edge:
            traderObject["volume_history"] = []
            return max(curr_edge - p["step_size"], p["min_edge"])
        return curr_edge

    def get_mid_price(self, depth: OrderDepth) -> float:
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2

    # Black-Scholes
    def norm_pdf(self, x): return math.exp(-0.5 * x*x)/math.sqrt(2*math.pi)
    def norm_cdf(self, x): return 0.5*(1+math.erf(x/math.sqrt(2)))
    def black_scholes_all(self, S, K, T, sigma):
        if T<=0: return 0,0,max(0,S-K)
        d1=(math.log(S/K)+(0.5*sigma**2)*T)/(sigma*math.sqrt(T))
        d2=d1-sigma*math.sqrt(T)
        delta=self.norm_cdf(d1)
        gamma=self.norm_pdf(d1)/(S*sigma*math.sqrt(T))
        price=S*self.norm_cdf(d1)-K*math.exp(-self.risk_free_rate*T)*self.norm_cdf(d2)
        return delta, gamma, price

    # -----------------------------------
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        # load or init state
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        # ----- 1) MACARONS trading & conversion -----
        mac = Product.MAGNIFICENT_MACARONS
        if mac in state.order_depths:
            # position and observation
            pos = state.position.get(mac, 0)
            obs: ConversionObservation = state.observations.conversionObservations[mac]
            depth: OrderDepth = state.order_depths[mac]
            # adaptive and biased edges
            edge = traderObject.get("curr_edge", self.params[mac]["init_make_edge"])
            edge = self.adaptive_edge(edge, pos, traderObject)
            traderObject["curr_edge"] = edge
            implied_bid, implied_ask = self.implied_bid_ask(obs)
            # sugar momentum
            s_hist = traderObject.setdefault("sugar_history", [])
            s_hist.append(obs.sugarPrice)
            if len(s_hist)>self.params[mac]["sugar_momentum_window"]: s_hist.pop(0)
            sugar_mom = (s_hist[-1]-s_hist[0]) if len(s_hist)==self.params[mac]["sugar_momentum_window"] else 0
            # base edges
            bid_edge = ask_edge = edge
            if sugar_mom>0: bid_edge=max(edge-self.params[mac]["sugar_bias_edge"],self.params[mac]["min_edge"])
            if sugar_mom<0: ask_edge=edge+self.params[mac]["sugar_bias_edge"]
            # sunlight bias
            if obs.sunlightIndex<self.params[mac]["CSI"]:
                bid_edge=max(bid_edge-self.params[mac]["sun_bias_edge"],self.params[mac]["min_edge"])
                ask_edge=ask_edge+self.params[mac]["sun_bias_edge"]
            # generate orders
            orders: List[Order]=[]
            limit=self.LIMIT[mac]
            # TAKE
            for a,pv in sorted(depth.sell_orders.items()):
                if a<=implied_bid-bid_edge:
                    q=min(pv, limit-pos)
                    orders.append(Order(mac,a,q)); pos+=q
            for b,pv in sorted(depth.buy_orders.items(), reverse=True):
                if b>=implied_ask+ask_edge:
                    q=min(pv, limit+pos)
                    orders.append(Order(mac,b,-q)); pos-=q
            # MAKE
            mbid=round(implied_bid-bid_edge)
            mask=round(implied_ask+ask_edge)
            if pos<limit: orders.append(Order(mac,mbid,limit-pos))
            if pos>-limit: orders.append(Order(mac,mask,-(limit+pos)))
            # CONVERSION (chunked)
            conv_lim=self.params[mac]["conv_limit"]
            to_conv=min(abs(pos), conv_lim)
            conv=int(-np.sign(pos)*to_conv) if pos!=0 else 0
            conversions = 0
            # record
            result[mac]=orders

        # ----- 2) Base commodities (Rainforest, Squid Ink, Kelp) -----
        base_products=["RAINFOREST_RESIN","SQUID_INK","KELP"]
        for prod in base_products:
            if prod not in state.order_depths: continue
            od=state.order_depths[prod]
            pos=self.position(state,prod)
            fair=self.get_fair_value(prod,od,traderObject)
            buy_v=sell_v=0
            orders=[]
            orders+=self.take_orders(prod,od,fair,pos,traderObject,buy_v,sell_v)
            orders+=self.clear_orders(prod,od,fair,pos,buy_v,sell_v)
            orders+=self.make_orders(prod,od,fair,pos,buy_v,sell_v)
            result[prod]=orders

        # ----- 3) Option strats on VOLCANIC_ROCK -----
        # update underlying
        ud=state.order_depths.get(self.underlying_symbol)
        mid_u=self.get_mid_price(ud)
        if mid_u is not None: self.last_prices[self.underlying_symbol]=mid_u
        else: mid_u=self.last_prices.get(self.underlying_symbol,10000)
        # total delta
        total_delta=0
        T=max(1,7-state.timestamp)/365
        # option quotes
        for sym,strike in self.option_symbols:
            od=state.order_depths.get(sym)
            if not od: continue
            mp=self.get_mid_price(od)
            if mp is None: continue
            self.last_prices[sym]=mp
            delta,gamma,theo=self.black_scholes_all(mid_u,strike,T,self.implied_volatility)
            total_delta+=delta*self.position(state,sym)
            opts=[]
            for a,pv in sorted(od.sell_orders.items()):
                if a<theo: opts.append(Order(sym,a,min(10,pv)))
            for b,pv in sorted(od.buy_orders.items(),reverse=True):
                if b>theo: opts.append(Order(sym,b,-min(10,pv)))
            result[sym]=opts
        # hedge underlying delta
        if ud:
            hedges=[]
            for b,pv in sorted(ud.buy_orders.items(),reverse=True):
                if total_delta<-9:
                    q=min(10,pv,int(-total_delta))
                    hedges.append(Order(self.underlying_symbol,b,-q)); total_delta+=q
            for a,pv in sorted(ud.sell_orders.items()):
                if total_delta>9:
                    q=min(10,pv,int(total_delta))
                    hedges.append(Order(self.underlying_symbol,a,q)); total_delta-=q
            result[self.underlying_symbol]=hedges

        # serialize state
        return result, conversions, jsonpickle.encode(traderObject)

    # Expose used helpers for base strat
    def position(self, state: TradingState, product: str) -> int:
        return state.position.get(product,0)
    def get_fair_value(self, product, od, td):
        from math import isnan
        if product=="RAINFOREST_RESIN": return 10000
        return self.adaptive_fair_value(product,od,td)
    def adaptive_fair_value(self, product, od, td):
        config={"SQUID_INK":{"adverse_volume":15,"reversion_beta":-0.25},
                "KELP":{"adverse_volume":20,"reversion_beta":-0.1}}
        if not od.sell_orders or not od.buy_orders:
            return td.get(f"{product}_last_price",10000)
        best_ask=min(od.sell_orders); best_bid=max(od.buy_orders)
        f_ask=[p for p,v in od.sell_orders.items() if abs(v)>=config[product]["adverse_volume"]]
        f_bid=[p for p,v in od.buy_orders.items() if abs(v)>=config[product]["adverse_volume"]]
        mm_ask=min(f_ask) if f_ask else None; mm_bid=max(f_bid) if f_bid else None
        if mm_ask is None or mm_bid is None:
            mmmid=td.get(f"{product}_last_price",(best_ask+best_bid)/2)
        else:
            mmmid=(mm_ask+mm_bid)/2
        last=td.get(f"{product}_last_price")
        if last:
            ret=(mmmid-last)/last
            mmmid+=mmmid*ret*config[product]["reversion_beta"]
        td[f"{product}_last_price"]=mmmid
        return mmmid
    # reuse make/take/clear from original
    def take_orders(self, product, od, fv, pos, td, bv, sv):
        # original take logic
        from datamodel import Order
        orders=[];limit=50
        cfg={"RAINFOREST_RESIN":{"take_width":0.8},
             "SQUID_INK":{"take_width":1,"prevent_adverse":True,"adverse_volume":15},
             "KELP":{"take_width":1.5,"prevent_adverse":True,"adverse_volume":20}}
        c=cfg[product]
        if od.sell_orders:
            best_ask=min(od.sell_orders);av=-od.sell_orders[best_ask]
            if not c.get("prevent_adverse",False) or av<=c.get("adverse_volume",0):
                if best_ask<=fv-c["take_width"]:
                    q=min(av,limit-pos)
                    if q>0: orders.append(Order(product,best_ask,q))
        if od.buy_orders:
            best_bid=max(od.buy_orders);bv=od.buy_orders[best_bid]
            if not c.get("prevent_adverse",False) or bv<=c.get("adverse_volume",0):
                if best_bid>=fv+c["take_width"]:
                    q=min(bv,limit+pos)
                    if q>0: orders.append(Order(product,best_bid,-q))
        return orders
    def clear_orders(self, product, od, fv, pos, bv, sv):
        from datamodel import Order
        orders=[];limit=50;cw=0
        p_after=pos+bv-sv
        bid_p=round(fv-cw);ask_p=round(fv+cw)
        buy_q=limit-(pos+bv);sell_q=limit+(pos-sv)
        if p_after>0:
            c=sum(v for p,v in od.buy_orders.items() if p>=ask_p)
            s=min(sell_q,min(c,p_after))
            if s>0: orders.append(Order(product,ask_p,-s))
        if p_after<0:
            c=sum(-v for p,v in od.sell_orders.items() if p<=bid_p)
            s=min(buy_q,min(c,-p_after))
            if s>0: orders.append(Order(product,bid_p,s))
        return orders
    def make_orders(self, product, od, fv, pos, bv, sv):
        from datamodel import Order
        orders=[];limit=50
        cfg={"RAINFOREST_RESIN":{"disregard_edge":1,"join_edge":1.5,"default_edge":3,"soft_position_limit":10},
             "SQUID_INK":{"disregard_edge":1,"join_edge":0,"default_edge":1},
             "KELP":{"disregard_edge":1.5,"join_edge":0.5,"default_edge":1.5}}
        c=cfg[product]
        asks=[p for p in od.sell_orders if p>fv+c["disregard_edge"]]
        bids=[p for p in od.buy_orders if p<fv-c["disregard_edge"]]
        ba=min(asks)-1 if asks and abs(min(asks)-fv)>c["join_edge"] else (min(asks) if asks else round(fv+c["default_edge"]))
        bb=max(bids)+1 if bids and abs(fv-max(bids))>c["join_edge"] else (max(bids) if bids else round(fv-c["default_edge"]))
        if pos>c.get("soft_position_limit",0): ba-=1
        if pos< -c.get("soft_position_limit",0): bb+=1
        bq=limit-(pos+bv)
        if bq>0: orders.append(Order(product,bb,bq))
        sq=limit+(pos-sv)
        if sq>0: orders.append(Order(product,ba,-sq))
        return orders