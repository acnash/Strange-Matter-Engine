Add-Type -AssemblyName System.Drawing
$figureDir = Join-Path $PSScriptRoot '../publication/figures'
$inputPath = [IO.Path]::GetFullPath((Join-Path $figureDir '02_five_rule_1d_cellular_evolution_before_label_edit.png'))
$outputPath = [IO.Path]::GetFullPath((Join-Path $figureDir '02_five_rule_1d_cellular_evolution_large_labels.png'))
$source = [Drawing.Bitmap]::FromFile($inputPath)
# Preserve the original raster at its native resolution; remove footer whitespace.
$crop = New-Object Drawing.Rectangle(0, 0, $source.Width, 1810)
$result = $source.Clone($crop, $source.PixelFormat)
$graphics = [Drawing.Graphics]::FromImage($result)
$graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$font = New-Object Drawing.Font('Arial', 39, [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
$brush = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(48, 54, 62))
$white = [Drawing.Brushes]::White
$format = New-Object Drawing.StringFormat
$format.Alignment = [Drawing.StringAlignment]::Center
$format.LineAlignment = [Drawing.StringAlignment]::Center
$format.FormatFlags = [Drawing.StringFormatFlags]::NoWrap
function Write-Number([string]$label, [double]$x, [double]$y) {
    $box = New-Object Drawing.RectangleF(($x - 52), ($y - 27), 104, 54)
    $graphics.DrawString($label, $font, $brush, $box, $format)
}
# Coordinates below use the original 3600 x 2040 figure, never a rescaled copy.
$graphics.FillRectangle($white, 120, 1540, 3430, 46)
$graphics.FillRectangle($white, 112, 1736, 3440, 61)
$panelLeft = @(150, 835, 1520, 2205, 2890)
$panelRight = @(770, 1455, 2140, 2825, 3510)
$counts = @(15, 23, 22, 15, 21)
$middle = @(8, 12, 11, 8, 11)
for ($panel = 0; $panel -lt 5; $panel++) {
    $left = $panelLeft[$panel]
    $right = $panelRight[$panel]
    $n = $counts[$panel]
    foreach ($atom in @(1, $middle[$panel], $n)) {
        $position = $left + ($atom - 0.5) / $n * ($right - $left)
        Write-Number "$atom" $position 1563
    }
    Write-Number ([string][char]0x2212 + '1') $left 1755
    Write-Number '0' (($left + $right) / 2) 1755
    Write-Number '1' $right 1755
}
$format.Alignment = [Drawing.StringAlignment]::Far
foreach ($generation in @(0, 64, 128, 192, 256)) {
    $y = 242 + $generation / 256 * (1525 - 242)
    $graphics.FillRectangle($white, 72, [int]($y - 25), 64, 50)
    $box = New-Object Drawing.RectangleF(50, ($y - 27), 84, 54)
    $graphics.DrawString([string]$generation, $font, $brush, $box, $format)
}
$graphics.Dispose()
# Verify every heatmap pixel is identical before saving the edited labels.
for ($panel = 0; $panel -lt 5; $panel++) {
    for ($x = $panelLeft[$panel]; $x -lt $panelRight[$panel]; $x++) {
        for ($y = 245; $y -lt 1524; $y++) {
            if ($source.GetPixel($x, $y).ToArgb() -ne $result.GetPixel($x, $y).ToArgb()) {
                throw "Scientific panel pixels changed at $x,$y"
            }
        }
    }
}
$result.Save($outputPath, [Drawing.Imaging.ImageFormat]::Png)
$source.Dispose()
$result.Dispose()
$font.Dispose()
$brush.Dispose()
$format.Dispose()
Write-Output "Saved $outputPath; all five heatmap pixel regions verified unchanged."
