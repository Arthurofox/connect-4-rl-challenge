# Connect-Four RL Challenge — Assignment Instructions

## Objective

Build an agent in Python that plays Connect-Four and **wins more often than your opponent’s agent**.

## What you must do

1. **Implement a Connect-Four RL agent** (any reasonable RL approach).
2. **Train locally on your own laptop** within a **shared, agreed time budget**.
3. **Keep the model lightweight** enough for smooth training on an M2 Mac.
4. **Freeze a final checkpoint** when your time budget ends.

## Match procedure

1. **Evaluation is head-to-head**: your final agent vs. your friend’s final agent.
2. **Play a fixed match** with an equal number of games starting first and second.
3. **No further training** during evaluation—only inference.

## Scoring

* **Primary:** overall win rate across the match.
* **If tied:** compare secondary stats (e.g., average moves to win, then model simplicity).

## Fair play

* Use only local compute.
* Use the same agreed training time window.
* Share only the frozen final agent for the match.

## What to hand in

* Your frozen agent.
* A brief note with the approach used and how long you trained.

That’s it. Train, submit, play the match, highest win rate wins.
