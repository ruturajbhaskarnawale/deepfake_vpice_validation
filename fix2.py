import re

with open('backend/app/templates/playground.html', 'r', encoding='utf-8') as f:
    html = f.read()

part1_start = html.find('<!-- Capture Modal Overlay -->')
script_start = html.find('<script>')
part2_start = html.find('<!-- Audio Recording Interface -->')

# The script tag ends right before part2
script_end_match = re.search(r'</script>\s*', html[script_start:])
script_end = script_start + script_end_match.end()

# Extract pieces
html_before = html[:part1_start]
modal_part1 = html[part1_start:script_start]
script_block = html[script_start:script_end]
modal_part2_and_rest = html[part2_start:]

# Split the very end (</body></html>) from modal_part2
body_end = modal_part2_and_rest.find('</body>')
modal_part2 = modal_part2_and_rest[:body_end]
html_after = modal_part2_and_rest[body_end:]

# Reconstruct
new_html = html_before + modal_part1 + modal_part2 + script_block + html_after

with open('backend/app/templates/playground.html', 'w', encoding='utf-8') as f:
    f.write(new_html)

print("Reconstructed!")
