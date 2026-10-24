from preprocessing import process_single_image
import os

img = r'datasets\P001\Rohith lower.jpg'
out = r'results\test_enhanced_pipeline'

print('Running enhanced pipeline on:', img)
summary, imp, dist, contours = process_single_image(img, output_dir=out)

if summary:
    print('SUCCESS')
    print(f'  Impressions detected : {summary["impression_count"]}')
    print(f'  Bite width           : {summary["bite_width"]}')
    print(f'  Avg spacing          : {summary["avg_spacing"]}')
    print(f'  Avg circularity      : {summary.get("avg_circularity", "N/A")}')
    print(f'  Debug images saved to: {out}')
    if os.path.exists(out):
        print(f'  Files: {sorted(os.listdir(out))}')
else:
    print('FAILED - returned None')
