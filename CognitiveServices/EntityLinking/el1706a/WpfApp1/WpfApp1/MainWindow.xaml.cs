using Microsoft.ProjectOxford.EntityLinking;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WpfApp1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private async void button_Click(object sender, RoutedEventArgs e)
        {
            var text = this.inputBox.Text;
            var client = new EntityLinkingServiceClient(System.Environment.GetEnvironmentVariable("el1706a_key1"));
            var linkResponse = await client.LinkAsync(text);
            var result = string.Join(", ", linkResponse.Select(i => i.WikipediaID).ToList());
            this.outputBlock.Text = result;
        }
    }
}
