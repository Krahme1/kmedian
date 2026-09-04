"""
Converts pmed k-median instances into k-facility instances by adding
random facility costs. Cost recipe: solve k-median for value V, then
each node's facility cost is random in [0, 2V/k].
"""
import json

# --- Step 1: load one existing pmed JSON and see what's inside ---
with open("tests/pmed1.json") as f:
    data = json.load(f)

n = data["n"]
k = data["k"]
distances = data["distances"]

print("Loaded pmed1.json")
print("  n (nodes)     :", n)
print("  k (facilities):", k)
print("  distances     : a", len(distances), "x", len(distances[0]), "matrix")
print("  distance[0][1]:", distances[0][1], "(distance between node 0 and node 1)")
