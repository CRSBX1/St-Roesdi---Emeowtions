import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Facebook, Instagram, Twitter } from "lucide-react";

const DocumentationDialog = ({ title, children }: { title: string, children: React.ReactNode }) => (
  <Dialog>
    <DialogTrigger asChild>
      <Button variant="ghost" className="text-background/80 hover:text-background hover:bg-background/10">
        {title}
      </Button>
    </DialogTrigger>
    <DialogContent className="max-w-2xl">
      <DialogHeader>
        <DialogTitle>{title}</DialogTitle>
      </DialogHeader>
      <div className="text-sm text-muted-foreground space-y-2">
        {children}
      </div>
    </DialogContent>
  </Dialog>
);

const Footer = () => {
  return (
    <footer className="bg-foreground text-background py-12 px-4">
      <div className="container mx-auto max-w-6xl">
        <div className="grid md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-2xl font-bold mb-4">Emeowtions</h3>
            <p className="text-background/80 mb-4">
              Understanding your cat's emotions through AI-powered analysis and personalized care recommendations.
            </p>
            <div className="flex space-x-4">
              <Button variant="ghost" size="icon" className="text-background hover:bg-background/10">
                <Facebook className="h-5 w-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-background hover:bg-background/10">
                <Instagram className="h-5 w-5" />
              </Button>
              <Button variant="ghost" size="icon" className="text-background hover:bg-background/10">
                <Twitter className="h-5 w-5" />
              </Button>
            </div>
          </div>
          <div>
            <h4 className="text-lg font-semibold mb-4">Documentation</h4>
            <div className="space-y-2">
              <DocumentationDialog title="How It Works">
                <p>Emeowtions analyzes your cat's audio recordings using advanced AI to detect emotional states.</p>
                <p>Simply upload a video (audio will be extracted), audio file, or record live audio of your cat.</p>
                <p>Our AI will analyze the vocalizations and provide insights about your cat's emotional state.</p>
                <p>You'll receive personalized recommendations on how to respond to your cat's needs.</p>
              </DocumentationDialog>
              
              <DocumentationDialog title="Privacy Policy">
                <p>Your cat's data and recordings are securely stored and never shared with third parties.</p>
                <p>All audio analysis is performed using encrypted connections.</p>
                <p>You can delete your data at any time from your account settings.</p>
                <p>We use industry-standard security measures to protect your information.</p>
              </DocumentationDialog>
              
              <DocumentationDialog title="Terms of Service">
                <p>By using Emeowtions, you agree to our terms of service.</p>
                <p>The AI analysis is for informational purposes and should not replace veterinary care.</p>
                <p>If your cat shows signs of illness, please consult a veterinarian.</p>
                <p>Refunds are available within 30 days of subscription purchase.</p>
              </DocumentationDialog>
              
              <DocumentationDialog title="Support">
                <p>Need help? We're here to assist you!</p>
                <p>Email us at: support@emeowtions.com</p>
                <p>Response time: Within 24 hours for premium users, 48 hours for free users.</p>
                <p>Check our FAQ section for common questions and troubleshooting tips.</p>
              </DocumentationDialog>
            </div>
          </div>
          <div>
            <h4 className="text-lg font-semibold mb-4">Contact</h4>
            <div className="space-y-2 text-background/80">
              <p>support@emeowtions.com</p>
              <p>1-800-MEOW-123</p>
              <p>Available 24/7 for premium users</p>
            </div>
          </div>
        </div>
        <div className="border-t border-background/20 mt-8 pt-8 text-center text-background/60">
          <p>&copy; 2024 Emeowtions. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;