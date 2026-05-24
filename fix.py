import re

with open('backend/app/templates/playground.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Find the modal
modal_match = re.search(r'<!-- Capture Modal Overlay -->.*?</div>\s*</div>\s*', html, re.DOTALL)
if modal_match:
    modal_html = modal_match.group(0)
    
    # 2. Remove it from the end
    html = html.replace(modal_html, '')
    
    # 3. Add help text to modal
    help_text = """
            <div id="recordingHelpText" style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); padding: 1rem; border-radius: 0.5rem; margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-primary); line-height: 1.4;">
                <strong style="color: var(--accent-cyan);">🗣️ Speaking Prompt:</strong><br>
                Please clearly state: <i>"Hi, my name is <full_name>, my date of birth is <date_of_birth>, my gender is <gender>, and my country of issuance is <country>."</i> This information will be cross-referenced with your uploaded documents to detect identity inconsistencies.
            </div>
"""
    new_modal = modal_html.replace('<!-- Video Recording Interface -->', help_text + '\n            <!-- Video Recording Interface -->')
    
    # 4. Insert modal before the main <script>
    html = html.replace('<script>', new_modal + '\n    <script>')
    
    with open('backend/app/templates/playground.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("Fixed successfully!")
else:
    print("Modal not found!")
