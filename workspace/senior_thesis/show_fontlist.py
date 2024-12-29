import matplotlib.font_manager as fm

# 利用可能なフォントの一覧を取得して表示
fonts = [f.name for f in fm.fontManager.ttflist]
fonts.sort()  # アルファベット順にソート

print("利用可能なフォント一覧:")
for font in fonts:
    print(f"- {font}")

# 日本語フォントをフィルタリングして表示（オプション）
japanese_fonts = [f for f in fonts if any(c > "\u4e00" and c < "\u9fff" for c in f)]
if japanese_fonts:
    print("\n日本語フォント:")
    for font in japanese_fonts:
        print(f"- {font}")
