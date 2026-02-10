//@version=6
strategy(
     "Forex QP Day Strategy (v6 – Strict QP Entry & Exit)",
     overlay = true,
     max_labels_count = 500,
     initial_capital = 100000,
     default_qty_type = strategy.fixed,
     default_qty_value = 1,
     process_orders_on_close = true
)

//----------------------------------------------------
// Inputs
//----------------------------------------------------
qpStep = input.float(0.250, "QP Step", step = 0.001)

//----------------------------------------------------
// Time Logic — IST
//----------------------------------------------------
isBatchCandle = (
    hour(time, "Asia/Kolkata") == 13 and
    minute(time, "Asia/Kolkata") == 30)

//----------------------------------------------------
// Helper
//----------------------------------------------------
f_qp(no) =>
    lower = math.floor(no / qpStep) * qpStep
    higher = lower + qpStep
    [lower, higher]

//----------------------------------------------------
// State Variables
//----------------------------------------------------
var float LQP = na
var float HQP = na

var float entry_price = na
var float target_price = na
var float stop_loss_price = na

var string direction = ""

var bool batch_active   = false
var bool entry_done     = false
var bool batch_complete = false

var int batch_start_bar = na

//----------------------------------------------------
// STEP 1 — Batch Start (13:30 IST)
//----------------------------------------------------
if isBatchCandle
    batch_active    := true
    entry_done      := false
    batch_complete  := false
    batch_start_bar := bar_index

    entry_price     := na
    target_price    := na
    stop_loss_price := na

    direction := close > open ? "BULLISH" : "BEARISH"
    [lqp_tmp, hqp_tmp] = f_qp(close)
    LQP := lqp_tmp
    HQP := hqp_tmp

    label.new(
        bar_index,
        high,
        "DIR\n" + direction,
        style = label.style_label_down,
        color = direction == "BULLISH" ? color.green : color.red,
        textcolor = color.white
    )


//----------------------------------------------------
// STEP 2 — ENTRY (SAME CANDLE, QP TOUCH ONLY)
//----------------------------------------------------
if batch_active and not entry_done and bar_index > batch_start_bar
    bool lqp_touch = low <= LQP and high >= LQP
    bool hqp_touch = low <= HQP and high >= HQP

    if direction == "BEARISH"
        if lqp_touch
            entry_price := LQP
        else if hqp_touch
            entry_price := HQP
    else
        if lqp_touch
            entry_price := LQP
        else if hqp_touch
            entry_price := HQP

    if not na(entry_price)
        entry_done := true

        target_price    := direction == "BULLISH" ? entry_price + qpStep : entry_price - qpStep
        stop_loss_price := direction == "BULLISH" ? entry_price - qpStep : entry_price + qpStep

        // ENTRY LABEL (QP)
        label.new(
            bar_index,
            entry_price,
            "ENTRY\n" + str.tostring(entry_price),
            style = label.style_label_left,
            color = color.blue,
            textcolor = color.white
        )

        // TARGET / SL LABELS (QP)
        label.new(bar_index, target_price,
            "TARGET\n" + str.tostring(target_price),
            style = label.style_label_down,
            color = color.green,
            textcolor = color.white,
            size = size.small)

        label.new(bar_index, stop_loss_price,
            "SL\n" + str.tostring(stop_loss_price),
            style = label.style_label_up,
            color = color.red,
            textcolor = color.white,
            size = size.small)

        // 🔒 ENTRY ORDER — STOP = LIMIT = ENTRY QP
        if direction == "BULLISH"
            strategy.entry(
                "LONG",
                strategy.long,
                stop  = entry_price,
                limit = entry_price
            )
        else
            strategy.entry(
                "SHORT",
                strategy.short,
                stop  = entry_price,
                limit = entry_price
            )


//----------------------------------------------------
// STEP 3 — EXIT (SAME CANDLE, QP TOUCH ONLY)
//----------------------------------------------------
if batch_active and entry_done and not batch_complete
    if direction == "BEARISH"
        // FAIL first (SL QP)
        if high >= stop_loss_price
            label.new(bar_index, stop_loss_price, "FAIL",
                style = label.style_label_left,
                color = color.red,
                textcolor = color.white)

            strategy.exit(
                "EXIT_SHORT_FAIL",
                from_entry = "SHORT",
                stop  = stop_loss_price,
                limit = stop_loss_price
            )
            batch_complete := true

        // PASS (TARGET QP)
        else if low <= target_price
            label.new(bar_index, target_price, "PASS",
                style = label.style_label_left,
                color = color.green,
                textcolor = color.white)

            strategy.exit(
                "EXIT_SHORT_PASS",
                from_entry = "SHORT",
                stop  = target_price,
                limit = target_price
            )
            batch_complete := true

    else
        // FAIL first
        if low <= stop_loss_price
            label.new(bar_index, stop_loss_price, "FAIL",
                style = label.style_label_left,
                color = color.red,
                textcolor = color.white)

            strategy.exit(
                "EXIT_LONG_FAIL",
                from_entry = "LONG",
                stop  = stop_loss_price,
                limit = stop_loss_price
            )
            batch_complete := true

        // PASS
        else if high >= target_price
            label.new(bar_index, target_price, "PASS",
                style = label.style_label_left,
                color = color.green,
                textcolor = color.white)

            strategy.exit(
                "EXIT_LONG_PASS",
                from_entry = "LONG",
                stop  = target_price,
                limit = target_price
            )
            batch_complete := true
