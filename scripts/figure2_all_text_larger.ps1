$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$dir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../publication/figures'))
$source = [Drawing.Bitmap]::FromFile((Join-Path $dir '02_five_rule_1d_cellular_evolution_before_label_edit.png'))
$result = New-Object Drawing.Bitmap(3660, 2020)
$g = [Drawing.Graphics]::FromImage($result)
$g.Clear([Drawing.Color]::White)
$g.CompositingMode = [Drawing.Drawing2D.CompositingMode]::SourceCopy
$g.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$ink = New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(48,54,62))
# At 174 mm printable width, two PDF points correspond to 14.84 pixels.
# Account also for the new 60-pixel left margin when preserving former sizes.
$increase = 2 * 3660 / (174 / 25.4 * 72)
$fonts = @{}
foreach($size in @(28,30,32,35,38,39,46)) {
    $fonts[$size] = New-Object Drawing.Font('Arial', ([single]($size * 3660/3600 + $increase)), [Drawing.FontStyle]::Regular, [Drawing.GraphicsUnit]::Pixel)
}
function Label([string]$text, [double]$x, [double]$y, [int]$size, [string]$align='Near') {
    $g.CompositingMode = [Drawing.Drawing2D.CompositingMode]::SourceOver
    $fmt = New-Object Drawing.StringFormat
    $fmt.Alignment = [Drawing.StringAlignment]::$align
    $fmt.LineAlignment = [Drawing.StringAlignment]::Center
    $fmt.FormatFlags = [Drawing.StringFormatFlags]::NoWrap
    $g.DrawString($text,$fonts[$size],$ink,(New-Object Drawing.PointF($x,$y)),$fmt)
    $fmt.Dispose()
}
$lefts = @(150,835,1520,2205,2890)
$titles = @('Gated residual',"Inertial`nreaction–diffusion",'FitzHugh–Nagumo','Kuramoto–Sakaguchi','Delayed memory')
$ids = @('OCNT-2328519','OCNT-2312382','OCNT-2314750','OCNT-2328784','OCNT-2308483')
$counts = @(15,23,22,15,21)
$mids = @(8,12,11,8,11)
for($i=0;$i -lt 5;$i++) {
    $x=$lefts[$i]; $newX=$x+60
    $plot=$source.Clone((New-Object Drawing.Rectangle(($x-10),240,640,1298)),$source.PixelFormat)
    $g.CompositingMode = [Drawing.Drawing2D.CompositingMode]::SourceCopy
    $g.DrawImage($plot,(New-Object Drawing.Rectangle(($newX-10),350,640,1298)),0,0,640,1298,[Drawing.GraphicsUnit]::Pixel)
    $plot.Dispose()
    $ramp=$source.Clone((New-Object Drawing.Rectangle($x,1706,620,26)),$source.PixelFormat)
    $g.DrawImage($ramp,(New-Object Drawing.Rectangle($newX,1850,620,26)),0,0,620,26,[Drawing.GraphicsUnit]::Pixel)
    $ramp.Dispose()
    Label ('('+[char](65+$i)+')') $newX 65 46
    $parts=$titles[$i].Split("`n")
    Label $parts[0] $newX 140 38
    if($parts.Length -gt 1){Label $parts[1] $newX 199 38}
    Label ($ids[$i]+' · CYP1A2') $newX 279 30
    foreach($n in @(1,$mids[$i],$counts[$i])) {
        Label "$n" ($newX+($n-.5)/$counts[$i]*620) 1690 39 'Center'
    }
    Label 'Atom index' ($newX+310) 1753 35 'Center'
    $key=if($i -eq 3){'Phase / π (channel 1)'}else{'State (channel 1)'}
    Label $key ($newX+310) 1810 32 'Center'
    Label '−1' $newX 1916 39 'Center'
    Label '0' ($newX+310) 1916 39 'Center'
    Label '1' ($newX+620) 1916 39 'Center'
}
foreach($n in @(0,64,128,192,256)){Label "$n" 193 (352+$n/256*1283) 39 'Far'}
$g.TranslateTransform(40,993)
$g.RotateTransform(-90)
Label 'Generation' 0 0 35 'Center'
$g.ResetTransform()
$g.Dispose()
Add-Type -ReferencedAssemblies System.Drawing.Common,System.Drawing.Primitives -TypeDefinition @'
using System; using System.Drawing;
public static class VerifyFigurePanels {
 public static void Check(Bitmap a, Bitmap b) {
  foreach(int left in new int[]{150,835,1520,2205,2890})
   for(int x=left;x<left+620;x++) for(int y=245;y<1524;y++)
    if(a.GetPixel(x,y).ToArgb()!=b.GetPixel(x+60,y+110).ToArgb())
     throw new Exception("Plot pixel changed");
 }
}
'@
[VerifyFigurePanels]::Check($source,$result)
$out=Join-Path $dir '02_five_rule_1d_cellular_evolution_all_text_plus2pt.png'
$result.Save($out,[Drawing.Imaging.ImageFormat]::Png)
$source.Dispose(); $result.Dispose(); $ink.Dispose()
foreach($font in $fonts.Values){$font.Dispose()}
Write-Output "Saved $out; all heatmap pixels verified unchanged."
